// ═══════════════════════════════════════════════════════════════════════
// Outred  - Frontend Application Logic
// Handles wizard steps, file upload, browser-mode detection (Z-score/IQR),
// server-mode API calls, and results rendering.
// ═══════════════════════════════════════════════════════════════════════

(function () {
    "use strict";

    // ── State ────────────────────────────────────────────────────────────
    let currentStep = 1;
    let processingMode = "browser"; // "browser" | "server"
    let uploadedFile = null;
    let parsedData = null; // { headers: string[], rows: any[][] }
    let analysisResults = null;

    // ── DOM refs ─────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const panels = {
        1: $("step-1-panel"),
        2: $("step-2-panel"),
        3: $("step-3-panel"),
    };

    // ── Wizard Navigation ────────────────────────────────────────────────
    function goToStep(n) {
        panels[currentStep].classList.add("hidden");
        panels[n].classList.remove("hidden");

        document.querySelectorAll(".step-indicator .step").forEach((el) => {
            const s = parseInt(el.dataset.step);
            el.classList.toggle("active", s === n);
            el.classList.toggle("completed", s < n);
        });

        currentStep = n;
    }

    // ── Mode Toggle ─────────────────────────────────────────────────────
    function setMode(mode) {
        processingMode = mode;
        document.querySelectorAll(".mode-btn").forEach((b) =>
            b.classList.toggle("active", b.dataset.mode === mode)
        );

        const badge = $("mode-badge");
        const privacy = $("privacy-notice");
        const privacyText = $("privacy-text");
        const serverOpts = $("server-options");

        if (mode === "browser") {
            badge.textContent = "Browser Mode";
            privacy.classList.remove("server-mode");
            privacyText.textContent =
                "Your data is processed entirely in your browser. Nothing is uploaded.";
            serverOpts.classList.add("hidden");
        } else {
            badge.textContent = "Server Mode";
            privacy.classList.add("server-mode");
            privacyText.textContent =
                "Data is uploaded for processing, then immediately deleted. Zero retention.";
            serverOpts.classList.remove("hidden");
        }
    }

    $("btn-browser-mode").addEventListener("click", () => setMode("browser"));
    $("btn-server-mode").addEventListener("click", () => setMode("server"));

    // ── File Upload ─────────────────────────────────────────────────────
    const dropZone = $("drop-zone");
    const fileInput = $("file-input");

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () =>
        dropZone.classList.remove("dragover")
    );
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        if (!file.name.toLowerCase().endsWith(".csv")) {
            alert("Please upload a CSV file.");
            return;
        }
        uploadedFile = file;
        $("file-info-name").textContent = file.name;
        $("file-info-size").textContent = formatBytes(file.size);
        $("file-info").classList.remove("hidden");
        $("drop-zone").classList.add("hidden");
        $("btn-to-step-2").classList.remove("hidden");
    }

    $("btn-change-file").addEventListener("click", () => {
        uploadedFile = null;
        parsedData = null;
        fileInput.value = "";
        $("file-info").classList.add("hidden");
        $("drop-zone").classList.remove("hidden");
        $("btn-to-step-2").classList.add("hidden");
    });

    $("btn-to-step-2").addEventListener("click", () => goToStep(2));
    $("btn-back-to-1").addEventListener("click", () => goToStep(1));

    // ── Sensitivity Slider ──────────────────────────────────────────────
    const slider = $("sensitivity-slider");
    const sliderVal = $("sensitivity-value");
    slider.addEventListener("input", () => {
        sliderVal.textContent = slider.value + "%";
    });

    // ── Run Analysis ────────────────────────────────────────────────────
    $("btn-run-analysis").addEventListener("click", runAnalysis);

    async function runAnalysis() {
        goToStep(3);
        $("loading-state").classList.remove("hidden");
        $("results-state").classList.add("hidden");
        $("loading-text").textContent = "Analyzing your data...";

        try {
            if (processingMode === "browser") {
                await runBrowserAnalysis();
            } else {
                await runServerAnalysis();
            }
        } catch (err) {
            $("loading-text").textContent = "Error: " + err.message;
        }
    }

    // ── Browser Analysis (Z-score + IQR) ────────────────────────────────
    async function runBrowserAnalysis() {
        $("loading-text").textContent = "Parsing CSV in browser...";

        const text = await uploadedFile.text();
        parsedData = parseCSV(text);

        $("loading-text").textContent = "Running outlier detection...";
        await sleep(50); // let the UI update

        const contamination = parseFloat(slider.value) / 100;
        const startTime = performance.now();

        // Identify numeric columns
        const numericIndices = [];
        const numericNames = [];
        for (let c = 0; c < parsedData.headers.length; c++) {
            const sample = parsedData.rows.slice(0, 20).map((r) => r[c]);
            if (sample.every((v) => v === null || v === "" || !isNaN(parseFloat(v)))) {
                numericIndices.push(c);
                numericNames.push(parsedData.headers[c]);
            }
        }

        if (numericIndices.length === 0) {
            throw new Error("No numeric columns found in the data.");
        }

        // Build numeric matrix
        const rows = parsedData.rows;
        const n = rows.length;

        // Compute column stats
        const stats = numericIndices.map((ci) => {
            const vals = rows.map((r) => parseFloat(r[ci])).filter((v) => !isNaN(v));
            vals.sort((a, b) => a - b);
            const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
            const std = Math.sqrt(
                vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length
            );
            const q1 = vals[Math.floor(vals.length * 0.25)];
            const q3 = vals[Math.floor(vals.length * 0.75)];
            const iqr = q3 - q1;
            return { mean, std: std || 1, q1, q3, iqr, median: vals[Math.floor(vals.length / 2)] };
        });

        // Score each row using combined Z-score + IQR method
        const scores = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            let maxScore = 0;
            for (let j = 0; j < numericIndices.length; j++) {
                const val = parseFloat(rows[i][numericIndices[j]]);
                if (isNaN(val)) continue;

                const s = stats[j];
                // Z-score component
                const zScore = Math.abs((val - s.mean) / s.std);
                // IQR component
                let iqrScore = 0;
                if (s.iqr > 0) {
                    const lower = s.q1 - 1.5 * s.iqr;
                    const upper = s.q3 + 1.5 * s.iqr;
                    if (val < lower) iqrScore = (lower - val) / s.iqr;
                    else if (val > upper) iqrScore = (val - upper) / s.iqr;
                }
                // Combined: weighted average
                const combined = zScore * 0.6 + iqrScore * 0.4;
                if (combined > maxScore) maxScore = combined;
            }
            scores[i] = maxScore;
        }

        // Normalize to 0–1
        let minS = Infinity, maxS = -Infinity;
        for (let i = 0; i < n; i++) {
            if (scores[i] < minS) minS = scores[i];
            if (scores[i] > maxS) maxS = scores[i];
        }
        const range = maxS - minS || 1;
        for (let i = 0; i < n; i++) {
            scores[i] = (scores[i] - minS) / range;
        }

        // Threshold
        const sorted = Array.from(scores).sort((a, b) => a - b);
        const threshold = sorted[Math.floor(n * (1 - contamination))];

        const outlierCount = scores.filter((s) => s >= threshold).length;
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        // Build results
        const outlierRows = [];
        for (let i = 0; i < n; i++) {
            if (scores[i] >= threshold) {
                const row = {};
                parsedData.headers.forEach((h, ci) => (row[h] = rows[i][ci]));
                row["anomaly_score"] = scores[i].toFixed(4);
                row["is_outlier"] = true;
                outlierRows.push(row);
            }
        }

        analysisResults = {
            summary: {
                total_rows: n,
                combined_outliers: outlierCount,
                numeric_outliers: outlierCount,
                categorical_outliers: 0,
                combined_pct: ((outlierCount / n) * 100).toFixed(2),
                numeric_pct: ((outlierCount / n) * 100).toFixed(2),
                categorical_pct: "0.00",
                runtime_seconds: parseFloat(elapsed),
                algorithm: "Z-Score + IQR (Browser)",
            },
            outliers: outlierRows.slice(0, 500),
            allScores: Array.from(scores),
            explanations: [], // Browser mode has no SHAP support
        };

        renderResults();
    }

    // ── Server Analysis ─────────────────────────────────────────────────
    async function runServerAnalysis() {
        $("loading-text").textContent = "Uploading to server...";

        const explainChecked = $("explain-toggle") && $("explain-toggle").checked;

        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("algorithm", $("algorithm-select").value);
        formData.append("contamination", parseFloat(slider.value) / 100);
        formData.append("scaling", $("scaling-select").value);
        formData.append("impute", $("impute-select").value);
        formData.append("explain", explainChecked ? "true" : "false");
        // Advanced settings
        formData.append("cat_threshold", $("adv-cat-threshold").value);
        formData.append("cat_max_cardinality", $("adv-cat-max-cardinality").value);
        formData.append("cat_min_cardinality", $("adv-cat-min-cardinality").value);
        formData.append("id_cardinality", $("adv-id-cardinality").value);
        formData.append("numeric_cast", $("adv-numeric-cast").value);
        formData.append("sample_rows", $("adv-sample-rows").value);
        formData.append("route_incremental_mb", $("adv-route-incremental-mb").value);
        formData.append("route_hbos_rows", $("adv-route-hbos-rows").value);
        formData.append("route_high_dims", $("adv-route-high-dims").value);
        formData.append("route_skewness", $("adv-route-skewness").value);

        // Large files can take a long time to upload + process — use a
        // generous 30-minute timeout so the browser doesn't abort early.
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30 * 60 * 1000);
        let res;
        try {
            res = await fetch("/api/analyze", { method: "POST", body: formData, signal: controller.signal });
        } finally {
            clearTimeout(timeoutId);
        }

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const detail = err.detail;
            // Structured validation error — show the issues panel instead of a
            // generic error message
            if (detail && typeof detail === "object" && Array.isArray(detail.issues)) {
                renderValidationErrors(detail);
                return;
            }
            throw new Error(
                (typeof detail === "string" ? detail : null) ||
                err.error ||
                `Server error ${res.status}`
            );
        }

        const data = await res.json();

        // Extract scores from outlier rows for histogram
        const allScores = data.outliers.map((r) => parseFloat(r.anomaly_score || 0));

        analysisResults = {
            summary: {
                ...data.summary,
                // Preserve the user's UI choice separately.
                // data.summary.algorithm = the actual algorithm the server ran.
                selected_algorithm: $("algorithm-select").value,
            },
            outliers: data.outliers,
            allScores,
            explanations: data.explanations || [],
        };

        renderResults();
    }

    // ── Render Results ──────────────────────────────────────────────────
    function renderResults() {
        $("loading-state").classList.add("hidden");
        $("validation-error-state").classList.add("hidden");
        $("results-state").classList.remove("hidden");

        const s = analysisResults.summary;
        $("card-total-rows").textContent = Number(s.total_rows).toLocaleString();
        $("card-outlier-count").textContent = Number(s.combined_outliers).toLocaleString();
        $("card-outlier-pct").textContent = s.combined_pct + "%";
        $("card-numeric").textContent = Number(s.numeric_outliers).toLocaleString();
        $("card-categorical").textContent = Number(s.categorical_outliers).toLocaleString();
        $("card-runtime").textContent = s.runtime_seconds + "s";
        // Algorithm card — show "Auto (HBOS)" when user picked auto
        const userChoice = s.selected_algorithm || "auto";
        const actualAlgo = (s.algorithm || userChoice).toUpperCase();
        $("card-algorithm").textContent = userChoice === "auto"
            ? `Auto (${actualAlgo})`
            : actualAlgo;

        renderHistogram(analysisResults.allScores);
        renderExplanations(analysisResults.explanations || []);
        renderTable(analysisResults.outliers);
    }

    // ── Validation Error Panel ──────────────────────────────────────────
    function renderValidationErrors(detail) {
        $("loading-state").classList.add("hidden");
        $("results-state").classList.add("hidden");
        $("validation-error-state").classList.remove("hidden");

        const total = detail.total_issues || detail.issues.length;
        $("validation-error-subtitle").textContent =
            `${total} structural issue${total !== 1 ? "s" : ""} found in your CSV. ` +
            `Fix the rows listed below, then re-upload the file.`;

        const list = $("validation-issues-list");
        list.innerHTML = "";

        for (const iss of detail.issues) {
            const item = document.createElement("div");
            item.className = "validation-issue-item";
            item.innerHTML = `
                <div class="vi-header">
                    <span class="vi-line">Line ${iss.line}</span>
                    <span class="vi-type">${escapeHtml(iss.type)}</span>
                </div>
                <div class="vi-desc">${escapeHtml(iss.description)}</div>
                <div class="vi-raw" title="${escapeHtml(iss.raw_content)}">${escapeHtml(iss.raw_content)}</div>
            `;
            list.appendChild(item);
        }

        // Wire download button — builds report text client-side, no extra request
        $("btn-download-validation").onclick = () => {
            const text = buildValidationReport(detail);
            const blob = new Blob([text], { type: "text/plain" });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href     = url;
            a.download = "outred_validation_report.txt";
            a.click();
            URL.revokeObjectURL(url);
        };
    }

    // Build a human-readable plain-text validation report from the JSON detail
    function buildValidationReport(detail) {
        const sep  = "=".repeat(60);
        const dash = "-".repeat(60);
        const lines = [];

        lines.push(sep);
        lines.push("  OUTRED -- CSV Validation Report");
        lines.push(sep);
        lines.push("");

        const total   = detail.total_issues || detail.issues.length;
        const showing = detail.issues.length;
        lines.push(`  Issues : ${total} structural issue${total !== 1 ? "s" : ""} found`);
        if (showing < total) {
            lines.push(`  Note   : Showing first ${showing} of ${total} issues`);
        }
        lines.push("");
        lines.push(dash);
        lines.push("");

        // Group consecutive issues by line number
        let lastLine = null;
        for (const iss of detail.issues) {
            if (iss.line !== lastLine) {
                if (lastLine !== null) lines.push("");
                lines.push(`[Line ${iss.line}]`);
                lastLine = iss.line;
            }
            lines.push(`  Type   : ${iss.type}`);
            lines.push(`  Reason : ${iss.description}`);
            lines.push(`  Raw    : ${iss.raw_content}`);
        }

        lines.push("");
        lines.push(dash);
        lines.push("  Please fix the rows listed above and re-upload your CSV.");
        lines.push(sep);

        return lines.join("\n");
    }

    $("btn-validation-retry").addEventListener("click", () => {
        uploadedFile = null;
        parsedData = null;
        analysisResults = null;
        fileInput.value = "";
        $("file-info").classList.add("hidden");
        $("drop-zone").classList.remove("hidden");
        $("btn-to-step-2").classList.add("hidden");
        $("validation-error-state").classList.add("hidden");
        goToStep(1);
    });

    // ── SHAP Explanations ───────────────────────────────────────────────
    function renderExplanations(explanations) {
        const container = $("explanations-container");
        const list = $("explanations-list");
        const countEl = $("explanations-count");

        list.innerHTML = "";

        if (!explanations || explanations.length === 0) {
            container.classList.add("hidden");
            return;
        }

        container.classList.remove("hidden");
        countEl.textContent = `(${explanations.length} row${explanations.length !== 1 ? "s" : ""} explained)`;

        // Find max absolute value across all for bar scaling
        let maxAbs = 0;
        for (const ex of explanations) {
            for (const f of ex.top_features) {
                if (Math.abs(f.value) > maxAbs) maxAbs = Math.abs(f.value);
            }
        }
        if (maxAbs === 0) maxAbs = 1;

        for (const ex of explanations) {
            const card = document.createElement("div");
            card.className = "explanation-card";

            const header = document.createElement("div");
            header.className = "explanation-card-header";
            header.innerHTML = `<span class="explanation-row-badge">Row ${ex.row_index}</span>`;
            card.appendChild(header);

            const featList = document.createElement("div");
            featList.className = "explanation-features";

            for (const f of ex.top_features) {
                const barPct = Math.min(100, (Math.abs(f.value) / maxAbs) * 100);
                const isPositive = f.value >= 0;
                const multiplier = f.median_value !== 0
                    ? (Math.abs(f.actual_value / f.median_value)).toFixed(1) + "×"
                    : "—";

                // Label differs: real SHAP vs z-score fallback
                const valueLabel = ex.method === "shap" ? "SHAP" : "Z-Score";
                const valueSign  = f.value > 0 ? "+" : "";

                const feat = document.createElement("div");
                feat.className = "explanation-feature";
                feat.innerHTML = `
                    <div class="feat-header">
                        <span class="feat-name">${escapeHtml(f.feature)}</span>
                        <span class="feat-multiplier">${multiplier} median</span>
                    </div>
                    <div class="feat-bar-track">
                        <div class="feat-bar ${isPositive ? "feat-bar-pos" : "feat-bar-neg"}" style="width:${barPct.toFixed(1)}%"></div>
                    </div>
                    <div class="feat-values">
                        <span class="feat-val-actual">Actual: <strong>${f.actual_value}</strong></span>
                        <span class="feat-val-median">Median: ${f.median_value}</span>
                        <span class="feat-shap">${valueLabel}: ${valueSign}${f.value}</span>
                    </div>
                `;
                featList.appendChild(feat);
            }

            card.appendChild(featList);
            list.appendChild(card);
        }
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    // ── Histogram (Canvas) ──────────────────────────────────────────────
    function renderHistogram(scores) {
        const canvas = $("score-histogram");
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (!scores || scores.length === 0) return;

        const bins = 30;
        const counts = new Array(bins).fill(0);
        for (const s of scores) {
            const idx = Math.min(Math.floor(s * bins), bins - 1);
            counts[idx]++;
        }
        const maxCount = Math.max(...counts, 1);

        const pad = { top: 20, right: 20, bottom: 35, left: 50 };
        const plotW = W - pad.left - pad.right;
        const plotH = H - pad.top - pad.bottom;
        const barW = plotW / bins;

        // Bars
        for (let i = 0; i < bins; i++) {
            const barH = (counts[i] / maxCount) * plotH;
            const x = pad.left + i * barW;
            const y = pad.top + plotH - barH;

            const frac = i / bins;
            const r = Math.round(99 + frac * 140);
            const g = Math.round(102 - frac * 60);
            const b = Math.round(241 - frac * 100);
            ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
            ctx.fillRect(x + 1, y, barW - 2, barH);
        }

        // Axes
        ctx.strokeStyle = "rgba(148,163,184,0.3)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.lineTo(pad.left + plotW, pad.top + plotH);
        ctx.stroke();

        // Labels
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("0.0", pad.left, pad.top + plotH + 18);
        ctx.fillText("0.5", pad.left + plotW / 2, pad.top + plotH + 18);
        ctx.fillText("1.0", pad.left + plotW, pad.top + plotH + 18);
        ctx.fillText("Anomaly Score", pad.left + plotW / 2, pad.top + plotH + 32);

        ctx.textAlign = "right";
        ctx.fillText(maxCount.toString(), pad.left - 6, pad.top + 10);
        ctx.fillText("0", pad.left - 6, pad.top + plotH + 4);
    }

    // ── Outlier Table ───────────────────────────────────────────────────
    function renderTable(outliers) {
        const thead = $("outlier-thead");
        const tbody = $("outlier-tbody");
        thead.innerHTML = "";
        tbody.innerHTML = "";

        if (!outliers || outliers.length === 0) {
            $("table-count").textContent = "(0 rows)";
            return;
        }

        $("table-count").textContent = `(showing ${Math.min(outliers.length, 200)} of ${outliers.length})`;

        // Headers
        const cols = Object.keys(outliers[0]);
        const trHead = document.createElement("tr");
        for (const col of cols) {
            const th = document.createElement("th");
            th.textContent = col;
            trHead.appendChild(th);
        }
        thead.appendChild(trHead);

        // Rows (cap at 200 for performance)
        const show = outliers.slice(0, 200);
        for (const row of show) {
            const tr = document.createElement("tr");
            tr.className = "outlier-row";
            for (const col of cols) {
                const td = document.createElement("td");
                const val = row[col];
                td.textContent = typeof val === "number" ? val.toFixed(4) : val;
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
    }

    // ── Download CSV ────────────────────────────────────────────────────
    $("btn-download-csv").addEventListener("click", () => {
        if (!analysisResults || !analysisResults.outliers.length) return;

        const rows = analysisResults.outliers;
        const cols = Object.keys(rows[0]);
        let csv = cols.join(",") + "\n";
        for (const row of rows) {
            csv += cols.map((c) => {
                const v = row[c];
                const s = String(v ?? "");
                return s.includes(",") ? `"${s}"` : s;
            }).join(",") + "\n";
        }

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "outred_results.csv";
        a.click();
        URL.revokeObjectURL(url);
    });

    // ── New Analysis ────────────────────────────────────────────────────
    $("btn-new-analysis").addEventListener("click", () => {
        uploadedFile = null;
        parsedData = null;
        analysisResults = null;
        fileInput.value = "";
        $("file-info").classList.add("hidden");
        $("drop-zone").classList.remove("hidden");
        $("btn-to-step-2").classList.add("hidden");
        goToStep(1);
    });

    // ── CSV Parser ──────────────────────────────────────────────────────
    function parseCSV(text) {
        const lines = text.trim().split("\n");
        if (lines.length < 2) throw new Error("CSV has no data rows.");

        const headers = splitCSVLine(lines[0]);
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            rows.push(splitCSVLine(line));
        }
        return { headers, rows };
    }

    function splitCSVLine(line) {
        const result = [];
        let current = "";
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') {
                inQuotes = !inQuotes;
            } else if (ch === "," && !inQuotes) {
                result.push(current.trim());
                current = "";
            } else {
                current += ch;
            }
        }
        result.push(current.trim());
        return result;
    }

    // ── Utils ───────────────────────────────────────────────────────────
    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function sleep(ms) {
        return new Promise((r) => setTimeout(r, ms));
    }
})();
