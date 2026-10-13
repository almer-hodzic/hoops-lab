const app = document.querySelector("#app");
const apiStatus = document.querySelector("#apiStatus");
const nav = document.querySelector("#mainNav");
const navToggle = document.querySelector("#navToggle");

const state = {
    model: null,
    selectedPlayer: null,
    historicalSelectedPlayer: null,
    historicalSelectedSeason: null,
    compareA: { player: null, season: null },
    compareB: { player: null, season: null },
    teamBuilder: null,
    teamProfiles: new Map(),
    teamEditor: null,
    selectedPoint: { x: 80, y: 250, distance: 26.2 },
    shotValue: 3,
    actionType: "Jump Shot",
    lastResult: null,
    historyPage: 1,
};

const court = {
    minX: -250,
    maxX: 250,
    minY: -52.5,
    maxY: 417.5,
    hoopX: 0,
    hoopY: 0,
    unitsPerFoot: 10,
    threeArcRadius: 237.5,
    cornerThreeX: 220,
    cornerBreakY: Math.sqrt((237.5 * 237.5) - (220 * 220)),
};

const facts = {
    shots: 1368990,
    clutch: 46613,
    seasons: "2020-21 to 2025-26",
    games: 7737,
    players: 1086,
    testSeason: "2025-26",
    regularSummary: [
        { decision: "TAKE", share: .201, xpps: 1.514, observed: 1.503 },
        { decision: "REVIEW", share: .542, xpps: 1.077, observed: 1.073 },
        { decision: "DEFER", share: .258, xpps: .803, observed: .808 },
    ],
    clutchSummary: [
        { decision: "TAKE", share: .156, observed: 1.508 },
        { decision: "REVIEW", share: .489, observed: 1.023 },
        { decision: "DEFER", share: .355, observed: .754 },
    ],
};

const modelMetrics = [
    ["2025-26 regular", "0.6665", "0.6358", "0.2234"],
    ["2025-26 clutch", "0.6818", "0.6241", "0.2178"],
    ["2025-26 playoffs", "0.6607", "0.6372", "0.2239"],
    ["2025-26 playoff clutch", "0.6945", "0.6128", "0.2129"],
];

function fmt(n, digits = 2) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "&mdash;";
    return Number(n).toFixed(digits);
}

function pct(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "&mdash;";
    return `${(Number(n) * 100).toFixed(digits)}%`;
}

function compact(n) {
    return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(n);
}

function pctPoint(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "&mdash;";
    return `${Number(n).toFixed(digits)}%`;
}

function signedPctPoint(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "&mdash;";
    const value = Number(n) * 100;
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}

function text(value) {
    if (value === null || value === undefined || value === "") return "&mdash;";
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function stat(label, value, note = "") {
    return `<div class="stat-row"><span>${label}</span><strong>${value}</strong>${note ? `<em>${note}</em>` : ""}</div>`;
}

function canonicalDistance(point) {
    return Math.sqrt((point.x * point.x) + (point.y * point.y)) / court.unitsPerFoot;
}

function modelShotDistance(point = state.selectedPoint) {
    return Math.floor(canonicalDistance(point));
}

function pointFromCanonical(x, y) {
    const clamped = {
        x: Math.max(court.minX, Math.min(court.maxX, x)),
        y: Math.max(court.minY, Math.min(court.maxY, y)),
    };
    return { ...clamped, distance: Number(canonicalDistance(clamped).toFixed(1)) };
}

function isThreePointLocation(point) {
    const absX = Math.abs(point.x);
    if (absX >= court.cornerThreeX && point.y <= court.cornerBreakY) return true;
    return canonicalDistance(point) * court.unitsPerFoot >= court.threeArcRadius && point.y > court.cornerBreakY;
}

function selectedShotValue() {
    return Number(document.querySelector("[data-shot-value].active")?.dataset.shotValue || state.shotValue || 3);
}

function shotValueWarning(point = state.selectedPoint, value = selectedShotValue()) {
    const locationIsThree = isThreePointLocation(point);
    if (value === 3 && !locationIsThree) return "Selected location is inside the NBA three-point line. Choose 2PT or move behind the arc.";
    if (value === 2 && locationIsThree) return "Selected location is behind the NBA three-point line. Choose 3PT or move inside the arc.";
    return "";
}

async function api(path, options = {}) {
    const response = await fetch(path, {
        headers: options.body ? { "Content-Type": "application/json" } : undefined,
        ...options,
    });
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

function setStatus(ok, text) {
    apiStatus.className = `status-dot ${ok ? "ok" : "bad"}`;
    apiStatus.textContent = text;
}

function pageName() {
    return (location.hash || "#overview").replace("#", "");
}

function setActiveNav() {
    const p = pageName();
    nav.querySelectorAll("a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === `#${p}`));
}

function panel(title, body, extra = "") {
    return `<section class="panel ${extra}">${title ? `<h3>${title}</h3>` : ""}${body}</section>`;
}

function metric(value, label) {
    return `<div class="metric"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}

function decisionBadge(decision) {
    return `<span class="decision-badge ${decision}">${decision}</span>`;
}

function bars(rows, key = "share") {
    return `<div class="bars-chart">${rows.map(r => {
        const color = r.decision === "TAKE" ? "var(--take)" : r.decision === "REVIEW" ? "var(--review)" : "var(--defer)";
        const width = Math.max(2, (r[key] || 0) * 100);
        return `<div class="bar-line"><strong>${r.decision}</strong><div class="track"><div class="fill" style="width:${width}%;background:${color}"></div></div><span>${key === "share" ? pct(r[key]) : fmt(r[key], 3)}</span></div>`;
    }).join("")}</div>`;
}

function renderOverview() {
    app.innerHTML = `
        <section class="hero">
            <div>
                <div class="eyebrow">Historical NBA analysis lab</div>
                <h1>HoopsLab</h1>
                <p class="lede">Explore historical player seasons, compare eras, and keep the shot decision lab close at hand.</p>
                <div class="hero-actions">
                    <a class="button" href="#players">Explore players</a>
                    <a class="button secondary" href="#lab">Open Shot Lab</a>
                </div>
                <div class="metrics-grid">
                    ${metric("20.5K", "historical player seasons")}
                    ${metric("1979-80", "first season")}
                    ${metric("2025-26", "latest season")}
                    ${metric("47", "NBA seasons")}
                </div>
            </div>
            <div class="panel">
                <div class="eyebrow">Product path</div>
                <div class="flow" style="margin-top:16px">
                    ${["Historical Players", "Season Profiles", "Team Builder", "Comparison", "Shot Lab"].map((x, i) => `<div class="flow-step"><span>0${i + 1}</span><strong>${x}</strong></div>`).join("")}
                </div>
                <div class="section">
                    <h3>Start with canonical data.</h3>
                    <p class="muted">The Historical Players module reads processed local HoopsLab datasets. It does not fabricate ratings or call live external sources.</p>
                </div>
            </div>
        </section>
        <section class="section">
            <div class="section-header">
                <div><h2>Shot decision intelligence stays available.</h2><p class="muted">The existing production shot model and decision policy remain unchanged.</p></div>
                <a class="button secondary" href="#compare">Compare players</a>
            </div>
            <div class="three-col">
                ${panel("Players", `<p class="muted">Search player identities and inspect every available season.</p>`)}
                ${panel("Compare", `<p class="muted">Put two historical player-seasons side by side without declaring an artificial winner.</p>`)}
                ${panel("Team Builder", `<p class="muted">Assemble a roster from specific historical player-seasons for future simulation work.</p>`)}
            </div>
        </section>`;
}

function courtSvg(point = state.selectedPoint) {
    const template = document.querySelector("#courtTemplate").content.cloneNode(true);
    const wrap = document.createElement("div");
    wrap.appendChild(template);
    const marker = wrap.querySelector("[data-shot-marker]");
    marker.setAttribute("transform", `translate(${point.x} ${point.y})`);
    return wrap.innerHTML;
}

function screenToCourt(svg, event) {
    const matrix = svg.getScreenCTM();
    if (!matrix) return state.selectedPoint;
    const pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const transformed = pt.matrixTransform(matrix.inverse());
    return pointFromCanonical(Math.round(transformed.x), Math.round(transformed.y));
}

function syncSelectedPoint(point, svg = document.querySelector(".court-svg")) {
    state.selectedPoint = point;
    const marker = svg?.querySelector("[data-shot-marker]");
    if (marker) marker.setAttribute("transform", `translate(${point.x} ${point.y})`);
    const distanceInput = document.querySelector("#shotDistance");
    const locXInput = document.querySelector("#locX");
    const locYInput = document.querySelector("#locY");
    if (distanceInput) distanceInput.value = point.distance;
    if (locXInput) locXInput.value = point.x;
    if (locYInput) locYInput.value = point.y;
    updateShotValueValidation();
    updateActionOptions();
}

function courtClick(event) {
    event.preventDefault();
    const svg = event.currentTarget;
    syncSelectedPoint(screenToCourt(svg, event), svg);
}

function setupCourtInteraction() {
    const svg = document.querySelector(".court-svg");
    if (!svg) return;
    svg.addEventListener("pointerdown", courtClick);
}

function updateShotValueValidation() {
    const box = document.querySelector("#shotValidation");
    if (!box) return;
    const warning = shotValueWarning();
    box.innerHTML = warning ? `<div class="notice">${warning}</div>` : `<div class="small">Location is consistent with the selected shot value.</div>`;
}

async function searchPlayers(query, targetId) {
    const box = document.querySelector(targetId);
    if (!query || query.length < 2) {
        box.innerHTML = "";
        return;
    }
    try {
        const data = await api(`/api/players?q=${encodeURIComponent(query)}&limit=8`);
        box.innerHTML = data.items.length ? data.items.map(p => `<button data-player='${JSON.stringify(p).replaceAll("'", "&apos;")}'>${p.player_name}<span class="small"> ${p.team_name || ""}</span></button>`).join("") : `<div class="small" style="padding:12px">No players found.</div>`;
        box.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
            state.selectedPlayer = JSON.parse(btn.dataset.player);
            document.querySelector("#playerSearch").value = state.selectedPlayer.player_name;
            box.innerHTML = "";
            loadSelectedProfile();
        }));
    } catch {
        box.innerHTML = `<div class="small" style="padding:12px">Player search unavailable.</div>`;
    }
}

function canonicalShotType(value) {
    return value === 3 ? "3PT Field Goal" : "2PT Field Goal";
}

function canonicalZone(point = state.selectedPoint, value = selectedShotValue()) {
    const x = Number(point.x);
    const y = Number(point.y);
    const distance = modelShotDistance(point);
    const backcourt = y > 417.5 || distance >= 40;
    let shotZoneRange;
    if (backcourt) shotZoneRange = "Back Court Shot";
    else if (value === 3) shotZoneRange = "24+ ft.";
    else if (distance < 8) shotZoneRange = "Less Than 8 ft.";
    else if (distance < 16) shotZoneRange = "8-16 ft.";
    else if (distance < 24) shotZoneRange = "16-24 ft.";
    else shotZoneRange = "24+ ft.";

    let shotZoneBasic;
    if (backcourt) shotZoneBasic = "Backcourt";
    else if (value === 3) shotZoneBasic = Math.abs(x) >= court.cornerThreeX && y <= 85 ? (x < 0 ? "Left Corner 3" : "Right Corner 3") : "Above the Break 3";
    else if (distance <= 3) shotZoneBasic = "Restricted Area";
    else if (Math.abs(x) <= 80 && y <= 140) shotZoneBasic = "In The Paint (Non-RA)";
    else shotZoneBasic = "Mid-Range";

    let shotZoneArea;
    if (backcourt) shotZoneArea = "Back Court(BC)";
    else if (Math.abs(x) <= 80) shotZoneArea = "Center(C)";
    else if (x < 0) shotZoneArea = y <= 80 ? "Left Side(L)" : "Left Side Center(LC)";
    else shotZoneArea = y <= 80 ? "Right Side(R)" : "Right Side Center(RC)";
    return { shotZoneBasic, shotZoneArea, shotZoneRange };
}

const shotActions = [
    { value: "Jump Shot", label: "Jump shot", group: "Jumpers" },
    { value: "Pullup Jump shot", label: "Pull-up jumper", group: "Jumpers" },
    { value: "Step Back Jump shot", label: "Step-back jumper", group: "Jumpers" },
    { value: "Fadeaway Jump Shot", label: "Fadeaway jumper", group: "Jumpers" },
    { value: "Turnaround Jump Shot", label: "Turnaround jumper", group: "Jumpers" },
    { value: "Floating Jump shot", label: "Floater", group: "Floaters" },
    { value: "Driving Floating Jump Shot", label: "Driving floater", group: "Floaters" },
    { value: "Layup Shot", label: "Layup", group: "Layups" },
    { value: "Driving Layup Shot", label: "Driving layup", group: "Layups" },
    { value: "Cutting Layup Shot", label: "Cutting layup", group: "Layups" },
    { value: "Finger Roll Layup Shot", label: "Finger roll", group: "Layups" },
    { value: "Hook Shot", label: "Hook", group: "Hooks" },
    { value: "Driving Hook Shot", label: "Driving hook", group: "Hooks" },
    { value: "Turnaround Hook Shot", label: "Turnaround hook", group: "Hooks" },
    { value: "Dunk Shot", label: "Dunk", group: "Dunks" },
    { value: "Driving Dunk Shot", label: "Driving dunk", group: "Dunks" },
    { value: "Cutting Dunk Shot", label: "Cutting dunk", group: "Dunks" },
];

function actionIsValid(actionType, value = selectedShotValue(), distance = modelShotDistance()) {
    const nearRim = /dunk|layup|hook/i.test(actionType);
    if (value === 3 && nearRim) return false;
    if (distance >= 16 && nearRim) return false;
    return true;
}

function validActionOptions(value = selectedShotValue(), distance = modelShotDistance()) {
    return shotActions.filter(a => actionIsValid(a.value, value, distance));
}

function updateActionOptions() {
    const select = document.querySelector("#actionType");
    if (!select) return;
    const value = selectedShotValue();
    const distance = modelShotDistance();
    const options = validActionOptions(value, distance);
    if (!options.some(o => o.value === state.actionType)) {
        state.actionType = options.some(o => o.value === "Jump Shot") ? "Jump Shot" : options[0].value;
    }
    const groups = [...new Set(options.map(o => o.group))];
    select.innerHTML = groups.map(group => {
        const items = options.filter(o => o.group === group).map(o => `<option value="${o.value}" ${o.value === state.actionType ? "selected" : ""}>${o.label}</option>`).join("");
        return `<optgroup label="${group}">${items}</optgroup>`;
    }).join("");
    const context = document.querySelector("#modelContext");
    if (context) {
        const zone = canonicalZone();
        context.innerHTML = `<div class="small">Model distance: ${distance} ft · ${canonicalShotType(value)} · ${zone.shotZoneBasic} / ${zone.shotZoneArea} / ${zone.shotZoneRange}</div>`;
    }
}

function derivedShotFields(value) {
    return {
        ...canonicalZone(state.selectedPoint, value),
        actionType: state.actionType,
        shotType: canonicalShotType(value),
        displayDistanceFeet: state.selectedPoint.distance,
        modelShotDistance: modelShotDistance(state.selectedPoint),
    };
}

async function analyzeShot() {
    const resultBox = document.querySelector("#resultBox");
    resultBox.innerHTML = `<div class="muted">Analyzing shot opportunity...</div>`;
    const value = Number(document.querySelector("[data-shot-value].active").dataset.shotValue);
    const warning = shotValueWarning(state.selectedPoint, value);
    if (warning) {
        resultBox.innerHTML = `<div class="error">${warning}</div>`;
        updateShotValueValidation();
        return;
    }
    state.actionType = document.querySelector("#actionType")?.value || state.actionType;
    if (!actionIsValid(state.actionType, value, modelShotDistance())) {
        resultBox.innerHTML = `<div class="error">${state.actionType} is not valid from this location.</div>`;
        updateActionOptions();
        return;
    }
    const scoreDiff = Number(document.querySelector("#scoreDiff").value);
    const fields = derivedShotFields(value);
    const payload = {
        playerId: state.selectedPlayer?.player_id || document.querySelector("#playerId").value || "999999999",
        playerName: state.selectedPlayer?.player_name,
        shotDistance: fields.modelShotDistance,
        displayDistanceFeet: fields.displayDistanceFeet,
        locX: Number(document.querySelector("#locX").value),
        locY: Number(document.querySelector("#locY").value),
        shotValue: value,
        ...fields,
        period: Number(document.querySelector("#period").value),
        secondsRemaining: Number(document.querySelector("#seconds").value),
        scoreDiff,
        isHome: document.querySelector("#homeAway").value === "home",
        isOvertime: Number(document.querySelector("#period").value) > 4,
        gameState: scoreDiff > 0 ? "leading" : scoreDiff < 0 ? "trailing" : "tied",
    };
    try {
        const result = await api("/api/decision/analyze", { method: "POST", body: JSON.stringify(payload) });
        state.lastResult = result;
        resultBox.innerHTML = resultHtml(result);
    } catch (err) {
        resultBox.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

function resultHtml(r) {
    const min = r.benchmark.replacementXPps - r.benchmark.threshold - .45;
    const max = r.benchmark.replacementXPps + r.benchmark.threshold + .45;
    const pos = Math.max(0, Math.min(100, ((r.prediction.shotXPps - min) / (max - min)) * 100));
    const bandLeft = ((r.benchmark.replacementXPps - r.benchmark.threshold - min) / (max - min)) * 100;
    const bandWidth = ((r.benchmark.threshold * 2) / (max - min)) * 100;
    const reason = {
        ABOVE_REPLACEMENT_VALUE: "Shot value is clearly above the replacement benchmark.",
        BELOW_REPLACEMENT_VALUE: "Shot value is below the replacement benchmark.",
        MARGIN_WITHIN_REVIEW_BAND: "Margin is inside the review band.",
        COLD_START_PROFILE: "Limited NBA history - league priors are used.",
    };
    return `
        <div class="recommendation ${r.recommendation}">${r.recommendation}</div>
        <p class="muted">${r.player.name} · ${(r.shot.displayDistanceFeet ?? r.shot.distance).toFixed(1)} ft display / ${r.shot.modelShotDistance ?? r.shot.distance} ft model · ${r.shot.actionType} · ${r.gameContext.gameState}</p>
        ${r.profile.coldStart ? `<div class="notice">Limited NBA history - league priors are used for this player profile.</div>` : ""}
        <div class="result-grid">
            <div class="mini-panel"><div class="label">Make probability</div><div class="num">${pct(r.prediction.makeProbability, 1)}</div></div>
            <div class="mini-panel"><div class="label">Shot expected value</div><div class="num">${fmt(r.prediction.shotXPps, 2)} xPPS</div></div>
            <div class="mini-panel"><div class="label">Replacement</div><div class="num">${fmt(r.benchmark.replacementXPps, 2)} xPPS</div></div>
            <div class="mini-panel"><div class="label">Decision margin</div><div class="num">${r.benchmark.margin >= 0 ? "+" : ""}${fmt(r.benchmark.margin, 2)}</div></div>
        </div>
        <div class="benchmark" aria-label="Shot value versus replacement benchmark">
            <div class="band" style="left:${bandLeft}%;width:${bandWidth}%"></div>
            <div class="marker" style="left:calc(${pos}% - 10px)"></div>
            <div class="small" style="display:flex;justify-content:space-between;margin-top:24px"><span>DEFER</span><span>REVIEW band</span><span>TAKE</span></div>
        </div>
        <div class="pill-row">${r.reasonCodes.map(x => `<span class="pill">${reason[x] || x}</span>`).join("")}</div>
        <div class="two-col">
            <div class="mini-panel"><div class="label">Zone profile</div><div class="num">${pct(r.profile.zonePriorFgPctShrunk, 1)}</div><p class="small">${r.profile.zonePriorAttempts} prior attempts in zone</p></div>
            <div class="mini-panel"><div class="label">Distance profile</div><div class="num">${pct(r.profile.distancePriorFgPctShrunk, 1)}</div><p class="small">${r.profile.distancePriorAttempts} prior attempts in bucket</p></div>
        </div>`;
}

async function loadSelectedProfile() {
    const box = document.querySelector("#profileContext");
    if (!box) return;
    if (!state.selectedPlayer) return;
    box.innerHTML = `<div class="muted">Loading profile...</div>`;
    try {
        const data = await api(`/api/players/${state.selectedPlayer.player_id}/profile`);
        const p = data.profiles?.[0];
        box.innerHTML = p ? `<div class="pill-row">
            <span class="pill">${p.team_name || "Team unavailable"}</span>
            <span class="pill">${p.player_prior_fga} prior FGA</span>
            <span class="pill">${pct(p.player_prior_fg_pct_shrunk, 1)} shrunk FG%</span>
            <span class="pill">${pct(p.player_prior_3p_pct_shrunk, 1)} shrunk 3P%</span>
        </div>` : `<div class="small">No profile detail available.</div>`;
    } catch (err) {
        box.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

function renderLab() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header">
                <div><div class="eyebrow">Shot Lab</div><h2>Analyze a shot opportunity.</h2><p class="muted">Click the court, choose game context, and call the production model.</p></div>
            </div>
            <div class="lab-grid">
                <div class="panel control-stack">
                    <div class="field autocomplete">
                        <label for="playerSearch">Player</label>
                        <input id="playerSearch" placeholder="Search player..." autocomplete="off" />
                        <input id="playerId" type="hidden" />
                        <div class="suggestions" id="playerSuggestions"></div>
                    </div>
                    <div id="profileContext" class="small">Select a player to view profile context.</div>
                    <div class="field"><label>Shot value</label><div class="segmented"><button data-shot-value="2">2PT</button><button data-shot-value="3" class="active">3PT</button></div></div>
                    <div id="shotValidation"></div>
                    <div class="two-col">
                        <div class="field"><label for="shotDistance">Distance</label><input id="shotDistance" type="number" value="${state.selectedPoint.distance}" min="0" max="100" step="0.1" readonly /></div>
                        <div class="field"><label for="period">Period</label><select id="period"><option value="4">Q4</option><option value="1">Q1</option><option value="2">Q2</option><option value="3">Q3</option><option value="5">OT</option></select></div>
                    </div>
                    <div class="field"><label for="actionType">Shot action</label><select id="actionType"></select></div>
                    <div id="modelContext"></div>
                    <div class="two-col">
                        <div class="field"><label for="seconds">Seconds remaining</label><input id="seconds" type="number" value="240" min="0" max="720" /></div>
                        <div class="field"><label for="scoreDiff">Score diff</label><input id="scoreDiff" type="number" value="-3" min="-80" max="80" /></div>
                    </div>
                    <div class="field"><label for="homeAway">Location</label><select id="homeAway"><option value="home">Home</option><option value="away">Away</option></select></div>
                    <input id="locX" type="hidden" value="${state.selectedPoint.x}" /><input id="locY" type="hidden" value="${state.selectedPoint.y}" />
                    <button class="button" id="analyzeButton">Analyze opportunity</button>
                </div>
                <div class="court-wrap">${courtSvg()}</div>
                <div class="panel result-panel" id="resultBox">
                    <div class="recommendation">READY</div>
                    <p class="muted">Select a player and analyze a location. The recommendation will appear here.</p>
                    <div class="notice">Production uses Decision Policy v1. Possession Policy v2 is research-only.</div>
                </div>
            </div>
        </section>`;
    setupCourtInteraction();
    document.querySelector("#playerSearch").addEventListener("input", e => searchPlayers(e.target.value, "#playerSuggestions"));
    document.querySelectorAll("[data-shot-value]").forEach(btn => btn.addEventListener("click", () => {
        document.querySelectorAll("[data-shot-value]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.shotValue = Number(btn.dataset.shotValue);
        updateShotValueValidation();
        updateActionOptions();
    }));
    document.querySelector("#actionType").addEventListener("change", e => { state.actionType = e.target.value; updateActionOptions(); });
    document.querySelector("#analyzeButton").addEventListener("click", analyzeShot);
    updateShotValueValidation();
    updateActionOptions();
}

async function searchHistoricalPlayers(query, targetId, onSelect) {
    const box = document.querySelector(targetId);
    if (!box) return;
    if (!query || query.length < 2) {
        box.innerHTML = "";
        return;
    }
    try {
        const data = await api(`/api/historical/players?search=${encodeURIComponent(query)}&limit=8`);
        box.innerHTML = data.items.length ? data.items.map(p => `
            <button data-player='${JSON.stringify(p).replaceAll("'", "&apos;")}'>
                ${text(p.playerName)}
                <span class="small">${text(p.firstSeason)} - ${text(p.lastSeason)} · ${p.numberOfSeasons} seasons</span>
            </button>`).join("") : `<div class="small" style="padding:12px">No historical players found.</div>`;
        box.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
            const player = JSON.parse(btn.dataset.player);
            box.innerHTML = "";
            onSelect(player);
        }));
    } catch (err) {
        box.innerHTML = `<div class="small" style="padding:12px">Historical player search unavailable.</div>`;
    }
}

function seasonOptions(profile, selectedSeason) {
    return profile.seasons.map(s => `<option value="${s.season}" ${s.season === selectedSeason ? "selected" : ""}>${s.season}</option>`).join("");
}

function metricCards(profile) {
    return `<div class="metrics-grid">
        ${metric(fmt(profile.scoring.pointsPer75, 1), "PTS / 75")}
        ${metric(fmt(profile.playmaking.assistsPer75, 1), "AST / 75")}
        ${metric(fmt(profile.rebounding.reboundsPer75, 1), "REB / 75")}
        ${metric(pct(profile.shooting.trueShootingPct, 1), "TS%")}
        ${metric(signedPctPoint(profile.shooting.relativeTrueShooting, 1), "Relative TS")}
        ${metric(pctPoint(profile.scoring.usagePct, 1), "USG%")}
    </div>`;
}

function eraBars(metrics) {
    const rows = Object.entries(metrics || {}).filter(([, m]) => m.seasonPercentile !== null && m.seasonPercentile !== undefined);
    if (!rows.length) return `<div class="small">No valid season-relative percentile data for this row.</div>`;
    return `<div class="era-bars">${rows.map(([label, m]) => `
        <div class="era-line">
            <div><strong>${text(label)}</strong><span>${fmt(m.seasonPercentile, 0)}th percentile</span></div>
            <div class="track"><div class="fill" style="width:${Math.max(2, Math.min(100, m.seasonPercentile))}%"></div></div>
        </div>`).join("")}</div>`;
}

function seasonProfileHtml(profile) {
    const i = profile.identity;
    return `
        <div class="profile-head">
            <div>
                <div class="eyebrow">Historical player profile</div>
                <h2>${text(i.playerName)}</h2>
                <p class="muted">${text(i.season)} · ${text(i.teamName)} · ${text(i.position)} · Age ${fmt(i.age, 0)}</p>
            </div>
            <div class="pill-row">
                <span class="pill">${text(i.team)}</span>
                <span class="pill">${fmt(i.games, 0)} games</span>
                <span class="pill">${fmt(i.minutes, 0)} minutes</span>
            </div>
        </div>
        ${metricCards(profile)}
        <div class="profile-grid">
            ${panel("Scoring", [
                stat("Points", fmt(profile.scoring.points, 0)),
                stat("PPG", fmt(profile.scoring.pointsPerGame, 1)),
                stat("PTS / 75", fmt(profile.scoring.pointsPer75, 1)),
                stat("USG%", pctPoint(profile.scoring.usagePct, 1)),
                stat("FGA", fmt(profile.scoring.fieldGoalAttempts, 0)),
                stat("3PAr", pct(profile.scoring.threePointAttemptRate, 1)),
            ].join(""))}
            ${panel("Shooting / efficiency", [
                stat("FG%", pct(profile.shooting.fieldGoalPct, 1)),
                stat("2P%", pct(profile.shooting.twoPointPct, 1)),
                stat("3P%", pct(profile.shooting.threePointPct, 1)),
                stat("FT%", pct(profile.shooting.freeThrowPct, 1)),
                stat("TS%", pct(profile.shooting.trueShootingPct, 1)),
                stat("League TS%", pct(profile.shooting.leagueTrueShootingPct, 1)),
                stat("Relative TS", signedPctPoint(profile.shooting.relativeTrueShooting, 1)),
                stat("FTr", fmt(profile.shooting.freeThrowRate, 3)),
            ].join(""))}
            ${panel("Playmaking", [
                stat("Assists", fmt(profile.playmaking.assists, 0)),
                stat("APG", fmt(profile.playmaking.assistsPerGame, 1)),
                stat("AST / 75", fmt(profile.playmaking.assistsPer75, 1)),
                stat("AST%", pctPoint(profile.playmaking.assistPct, 1)),
                stat("Turnovers", fmt(profile.playmaking.turnovers, 0)),
                stat("TOV / 75", fmt(profile.playmaking.turnoversPer75, 1)),
                stat("TOV%", pctPoint(profile.playmaking.turnoverPct, 1)),
            ].join(""))}
            ${panel("Rebounding", [
                stat("ORB", fmt(profile.rebounding.offensiveRebounds, 0)),
                stat("DRB", fmt(profile.rebounding.defensiveRebounds, 0)),
                stat("TRB", fmt(profile.rebounding.totalRebounds, 0)),
                stat("RPG", fmt(profile.rebounding.reboundsPerGame, 1)),
                stat("REB / 75", fmt(profile.rebounding.reboundsPer75, 1)),
                stat("TRB%", pctPoint(profile.rebounding.totalReboundPct, 1)),
            ].join(""))}
            ${panel("Defense", [
                stat("Steals", fmt(profile.defense.steals, 0)),
                stat("SPG", fmt(profile.defense.stealsPerGame, 1)),
                stat("Steals / 75", fmt(profile.defense.stealsPer75, 1)),
                stat("STL%", profile.defense.stealPct === null ? "Not tracked" : pctPoint(profile.defense.stealPct, 1)),
                stat("Blocks", fmt(profile.defense.blocks, 0)),
                stat("BPG", fmt(profile.defense.blocksPerGame, 1)),
                stat("Blocks / 75", fmt(profile.defense.blocksPer75, 1)),
                stat("BLK%", profile.defense.blockPct === null ? "Not tracked" : pctPoint(profile.defense.blockPct, 1)),
            ].join(""))}
            ${panel("Source-derived advanced", [
                stat("PER", fmt(profile.advanced.playerEfficiencyRating, 1)),
                stat("WS", fmt(profile.advanced.winShares, 1)),
                stat("WS / 48", profile.advanced.winSharesPer48 === null ? "Not tracked" : fmt(profile.advanced.winSharesPer48, 3)),
                stat("BPM", fmt(profile.advanced.boxPlusMinus, 1)),
                stat("OBPM", profile.advanced.offensiveBoxPlusMinus === null ? "Not tracked" : fmt(profile.advanced.offensiveBoxPlusMinus, 1)),
                stat("DBPM", profile.advanced.defensiveBoxPlusMinus === null ? "Not tracked" : fmt(profile.advanced.defensiveBoxPlusMinus, 1)),
                stat("VORP", profile.advanced.valueOverReplacementPlayer === null ? "Not tracked" : fmt(profile.advanced.valueOverReplacementPlayer, 1)),
            ].join("") + `<p class="small">${text(profile.advanced.provenance)}</p>`)}
            ${panel("Era context", [
                stat("Team pace", fmt(profile.eraContext.teamPace, 1)),
                stat("League pace", fmt(profile.eraContext.leaguePace, 1)),
                stat("League points / 75", fmt(profile.eraContext.leaguePointsPer75, 1)),
                stat("Estimated possessions", fmt(profile.eraContext.estimatedPossessions, 0)),
                stat("Sample flag", text(profile.eraContext.comparisonSample)),
            ].join("") + eraBars(profile.eraContext.metrics))}
        </div>`;
}

async function loadHistoricalSeason(playerId, season, targetId = "#historicalProfile") {
    const target = document.querySelector(targetId);
    target.innerHTML = `<div class="muted">Loading season...</div>`;
    try {
        const profile = await api(`/api/historical/players/${encodeURIComponent(playerId)}/seasons/${encodeURIComponent(season)}`);
        target.innerHTML = seasonProfileHtml(profile);
    } catch (err) {
        target.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

async function openHistoricalPlayer(player) {
    state.historicalSelectedPlayer = player;
    const profileBox = document.querySelector("#historicalProfile");
    profileBox.innerHTML = `<div class="muted">Loading ${text(player.playerName)}...</div>`;
    const profile = await api(`/api/historical/players/${encodeURIComponent(player.playerId)}`);
    state.historicalSelectedSeason = profile.seasons[profile.seasons.length - 1].season;
    profileBox.innerHTML = `
        <div class="panel">
            <div class="profile-toolbar">
                <div>
                    <h3>${text(profile.playerName)}</h3>
                    <p class="muted">${text(profile.firstSeason)} - ${text(profile.lastSeason)} · ${profile.numberOfSeasons} seasons</p>
                </div>
                <div class="field season-picker"><label for="historicalSeason">Season</label><select id="historicalSeason">${seasonOptions(profile, state.historicalSelectedSeason)}</select></div>
            </div>
            <div id="historicalSeasonProfile"></div>
        </div>`;
    document.querySelector("#historicalSeason").addEventListener("change", e => {
        state.historicalSelectedSeason = e.target.value;
        loadHistoricalSeason(profile.playerId, state.historicalSelectedSeason, "#historicalSeasonProfile");
    });
    await loadHistoricalSeason(profile.playerId, state.historicalSelectedSeason, "#historicalSeasonProfile");
}

async function renderPlayers() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Historical Players</div><h2>Explore NBA player seasons.</h2><p class="muted">Search the canonical HoopsLab historical dataset from 1979-80 through 2025-26.</p></div></div>
            <div class="historical-layout">
                <div class="panel">
                    <div class="field autocomplete"><label for="historicalPlayerSearch">Search players</label><input id="historicalPlayerSearch" placeholder="Try Jordan, Curry, LeBron, Jokic..." autocomplete="off" /><div class="suggestions" id="historicalPlayerSuggestions"></div></div>
                    <div class="notice" style="margin-top:16px">Historical profiles use processed local CSV outputs only. No live Basketball-Reference request is made by the app.</div>
                </div>
                <div id="historicalProfile" class="historical-profile">
                    <div class="panel empty-state"><h3>Select a player</h3><p class="muted">Open a profile, choose a season, and inspect raw, per-possession, and era-relative stats.</p></div>
                </div>
            </div>
        </section>`;
    document.querySelector("#historicalPlayerSearch").addEventListener("input", e => searchHistoricalPlayers(e.target.value, "#historicalPlayerSuggestions", player => {
        document.querySelector("#historicalPlayerSearch").value = player.playerName;
        openHistoricalPlayer(player);
    }));
}

async function setComparePlayer(side, player) {
    state[side].player = player;
    state[side].season = null;
    const label = side === "compareA" ? "A" : "B";
    document.querySelector(`#${side}Search`).value = player.playerName;
    const seasonBox = document.querySelector(`#${side}SeasonWrap`);
    seasonBox.innerHTML = `<div class="muted">Loading seasons...</div>`;
    const profile = await api(`/api/historical/players/${encodeURIComponent(player.playerId)}`);
    state[side].season = profile.seasons[profile.seasons.length - 1].season;
    seasonBox.innerHTML = `<label for="${side}Season">Season ${label}</label><select id="${side}Season">${seasonOptions(profile, state[side].season)}</select>`;
    document.querySelector(`#${side}Season`).addEventListener("change", e => {
        state[side].season = e.target.value;
        loadComparison();
    });
    await loadComparison();
}

function compareStat(label, getter, formatter = value => fmt(value, 1)) {
    const a = getter(state.compareResult?.playerA);
    const b = getter(state.compareResult?.playerB);
    return `<tr><td>${label}</td><td>${formatter(a)}</td><td>${formatter(b)}</td></tr>`;
}

function comparisonTable(title, rows) {
    return panel(title, `<div class="table-wrap comparison-table"><table><thead><tr><th>Metric</th><th>${text(state.compareResult.playerA.identity.playerName)}<br><span>${text(state.compareResult.playerA.identity.season)}</span></th><th>${text(state.compareResult.playerB.identity.playerName)}<br><span>${text(state.compareResult.playerB.identity.season)}</span></th></tr></thead><tbody>${rows}</tbody></table></div>`);
}

async function loadComparison() {
    const target = document.querySelector("#comparisonResult");
    if (!target) return;
    const a = state.compareA;
    const b = state.compareB;
    if (!a.player || !a.season || !b.player || !b.season) {
        target.innerHTML = `<div class="panel empty-state"><h3>Choose two player-seasons</h3><p class="muted">The comparison keeps raw, per-75, and era-relative values side by side without declaring a winner.</p></div>`;
        return;
    }
    target.innerHTML = `<div class="muted">Loading comparison...</div>`;
    try {
        state.compareResult = await api(`/api/historical/compare?playerA=${encodeURIComponent(a.player.playerId)}&seasonA=${encodeURIComponent(a.season)}&playerB=${encodeURIComponent(b.player.playerId)}&seasonB=${encodeURIComponent(b.season)}`);
        const result = state.compareResult;
        target.innerHTML = `
            <div class="compare-head panel">
                <div><h2>${text(result.playerA.identity.playerName)}</h2><p class="muted">${text(result.playerA.identity.season)} · ${text(result.playerA.identity.teamName)}</p></div>
                <div class="versus">vs</div>
                <div><h2>${text(result.playerB.identity.playerName)}</h2><p class="muted">${text(result.playerB.identity.season)} · ${text(result.playerB.identity.teamName)}</p></div>
            </div>
            <div class="profile-grid">
                ${comparisonTable("Scoring", [
                    compareStat("Points", p => p?.scoring.points, v => fmt(v, 0)),
                    compareStat("PPG", p => p?.scoring.pointsPerGame),
                    compareStat("PTS / 75", p => p?.scoring.pointsPer75),
                    compareStat("USG%", p => p?.scoring.usagePct, v => pctPoint(v, 1)),
                    compareStat("FGA", p => p?.scoring.fieldGoalAttempts, v => fmt(v, 0)),
                ].join(""))}
                ${comparisonTable("Efficiency", [
                    compareStat("FG%", p => p?.shooting.fieldGoalPct, pct),
                    compareStat("2P%", p => p?.shooting.twoPointPct, pct),
                    compareStat("3P%", p => p?.shooting.threePointPct, pct),
                    compareStat("FT%", p => p?.shooting.freeThrowPct, pct),
                    compareStat("TS%", p => p?.shooting.trueShootingPct, pct),
                    compareStat("Relative TS", p => p?.shooting.relativeTrueShooting, signedPctPoint),
                    compareStat("3PAr", p => p?.shooting.threePointAttemptRate, pct),
                    compareStat("FTr", p => p?.shooting.freeThrowRate, v => fmt(v, 3)),
                ].join(""))}
                ${comparisonTable("Playmaking", [
                    compareStat("Assists", p => p?.playmaking.assists, v => fmt(v, 0)),
                    compareStat("AST / 75", p => p?.playmaking.assistsPer75),
                    compareStat("AST%", p => p?.playmaking.assistPct, v => pctPoint(v, 1)),
                    compareStat("Turnovers", p => p?.playmaking.turnovers, v => fmt(v, 0)),
                    compareStat("TOV / 75", p => p?.playmaking.turnoversPer75),
                    compareStat("TOV%", p => p?.playmaking.turnoverPct, v => pctPoint(v, 1)),
                ].join(""))}
                ${comparisonTable("Rebounding / defense", [
                    compareStat("REB / 75", p => p?.rebounding.reboundsPer75),
                    compareStat("TRB%", p => p?.rebounding.totalReboundPct, v => pctPoint(v, 1)),
                    compareStat("Steals / 75", p => p?.defense.stealsPer75),
                    compareStat("Blocks / 75", p => p?.defense.blocksPer75),
                ].join(""))}
                ${comparisonTable("Advanced", [
                    compareStat("PER", p => p?.advanced.playerEfficiencyRating),
                    compareStat("WS", p => p?.advanced.winShares),
                    compareStat("BPM", p => p?.advanced.boxPlusMinus),
                ].join(""))}
                ${comparisonTable("Era context", [
                    compareStat("PTS / 75 percentile", p => p?.eraContext.metrics["Points / 75"]?.seasonPercentile, v => fmt(v, 0)),
                    compareStat("Relative TS percentile", p => p?.eraContext.metrics["Relative TS"]?.seasonPercentile, v => fmt(v, 0)),
                    compareStat("AST% percentile", p => p?.eraContext.metrics["AST%"]?.seasonPercentile, v => fmt(v, 0)),
                    compareStat("TRB% percentile", p => p?.eraContext.metrics["TRB%"]?.seasonPercentile, v => fmt(v, 0)),
                    compareStat("Team pace", p => p?.eraContext.teamPace),
                    compareStat("League pace", p => p?.eraContext.leaguePace),
                ].join(""))}
            </div>`;
    } catch (err) {
        target.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

async function renderCompare() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Compare</div><h2>Historical player-season comparison.</h2><p class="muted">Choose two seasons and compare real historical production, efficiency, and era context.</p></div></div>
            <div class="compare-controls">
                <div class="panel control-stack">
                    <div class="field autocomplete"><label for="compareASearch">Player A</label><input id="compareASearch" placeholder="Michael Jordan..." autocomplete="off" /><div class="suggestions" id="compareASuggestions"></div></div>
                    <div class="field" id="compareASeasonWrap"><label>Season A</label><select disabled><option>Select player first</option></select></div>
                </div>
                <div class="panel control-stack">
                    <div class="field autocomplete"><label for="compareBSearch">Player B</label><input id="compareBSearch" placeholder="Stephen Curry..." autocomplete="off" /><div class="suggestions" id="compareBSuggestions"></div></div>
                    <div class="field" id="compareBSeasonWrap"><label>Season B</label><select disabled><option>Select player first</option></select></div>
                </div>
            </div>
            <div id="comparisonResult" style="margin-top:16px"></div>
        </section>`;
    document.querySelector("#compareASearch").addEventListener("input", e => searchHistoricalPlayers(e.target.value, "#compareASuggestions", player => setComparePlayer("compareA", player)));
    document.querySelector("#compareBSearch").addEventListener("input", e => searchHistoricalPlayers(e.target.value, "#compareBSuggestions", player => setComparePlayer("compareB", player)));
    loadComparison();
}

function team() {
    if (!state.teamBuilder) state.teamBuilder = TeamBuilder.load();
    return state.teamBuilder;
}

function setTeam(next) {
    state.teamBuilder = next;
    TeamBuilder.save(next);
}

function entryKey(entry) {
    return `${entry.playerId}|${entry.season}`;
}

async function profileForEntry(entry) {
    if (!entry) return null;
    const key = entryKey(entry);
    if (!state.teamProfiles.has(key)) {
        state.teamProfiles.set(key, await api(`/api/historical/players/${encodeURIComponent(entry.playerId)}/seasons/${encodeURIComponent(entry.season)}`));
    }
    return state.teamProfiles.get(key);
}

function positionWarning(entry, slot) {
    if (!entry || slot.startsWith("BENCH")) return "";
    const position = entry.position || "";
    if (!position || position.split("-").includes(slot)) return "";
    return `<div class="small subtle-warning">${text(position)} listed position in ${text(entry.season)}. Placement is allowed.</div>`;
}

async function rosterCard(entry, slot) {
    if (!entry) {
        return `<div class="roster-card empty">
            <div><strong>${slot}</strong><span>Empty slot</span></div>
            <button class="button secondary" data-team-add="${slot}">Add Player</button>
        </div>`;
    }
    const profile = await profileForEntry(entry);
    return `<div class="roster-card">
        <div class="roster-main">
            <div class="slot-tag">${slot}</div>
            <div>
                <h3>${text(entry.playerName)}</h3>
                <p class="muted">${text(entry.season)} · ${text(entry.team)} · ${text(entry.position)}</p>
                ${positionWarning(entry, slot)}
            </div>
        </div>
        <div class="roster-stats">
            ${metric(fmt(profile?.scoring.pointsPer75, 1), "PTS / 75")}
            ${metric(fmt(profile?.playmaking.assistsPer75, 1), "AST / 75")}
            ${metric(fmt(profile?.rebounding.reboundsPer75, 1), "REB / 75")}
            ${metric(pct(profile?.shooting.trueShootingPct, 1), "TS%")}
            ${metric(signedPctPoint(profile?.shooting.relativeTrueShooting, 1), "Rel TS")}
        </div>
        <div class="roster-actions">
            <button class="button secondary" data-team-detail="${slot}">Details</button>
            <button class="button secondary" data-team-replace="${slot}">Replace</button>
            <button class="button secondary" data-team-season="${slot}">Change season</button>
            <select data-team-move="${slot}" aria-label="Move ${text(entry.playerName)}">
                <option value="">Move to...</option>
                ${TeamBuilder.STARTER_SLOTS.map(s => `<option value="${s}">${s}</option>`).join("")}
                <option value="BENCH">Bench</option>
            </select>
            <button class="button secondary" data-team-remove="${slot}">Remove</button>
        </div>
    </div>`;
}

function decadeLabel(season) {
    const start = Number(String(season).slice(0, 4));
    if (!Number.isFinite(start)) return "Unknown";
    return `${Math.floor(start / 10) * 10}s`;
}

async function teamSummaryHtml(current) {
    const entries = TeamBuilder.rosterEntries(current);
    const profiles = await Promise.all(entries.map(profileForEntry));
    const ages = profiles.map(p => p?.identity.age).filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)));
    const pts = profiles.map(p => p?.scoring.pointsPer75).filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)));
    const ast = profiles.map(p => p?.playmaking.assistsPer75).filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)));
    const reb = profiles.map(p => p?.rebounding.reboundsPer75).filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)));
    const avg = arr => arr.length ? arr.reduce((a, b) => a + Number(b), 0) / arr.length : null;
    const eraCounts = entries.reduce((acc, entry) => {
        const label = decadeLabel(entry.season);
        acc[label] = (acc[label] || 0) + 1;
        return acc;
    }, {});
    const maxEra = Math.max(1, ...Object.values(eraCounts));
    return `
        <div class="metrics-grid team-summary">
            ${metric(`${TeamBuilder.rosterSize(current)} / 12`, "roster size")}
            ${metric(fmt(avg(ages), 1), "average age")}
            ${metric(fmt(avg(pts), 1), "avg PTS / 75")}
            ${metric(fmt(avg(ast), 1), "avg AST / 75")}
            ${metric(fmt(avg(reb), 1), "avg REB / 75")}
        </div>
        <div class="panel">
            <h3>Era distribution</h3>
            <div class="era-bars">${Object.entries(eraCounts).sort().map(([label, count]) => `
                <div class="era-line"><div><strong>${label}</strong><span>${count}</span></div><div class="track"><div class="fill" style="width:${Math.max(8, count / maxEra * 100)}%"></div></div></div>`).join("") || `<div class="small">No players selected yet.</div>`}</div>
        </div>`;
}

async function renderRoster() {
    const current = team();
    document.querySelector("#teamSummary").innerHTML = await teamSummaryHtml(current);
    document.querySelector("#startersGrid").innerHTML = (await Promise.all(TeamBuilder.STARTER_SLOTS.map(slot => rosterCard(current.starters[slot], slot)))).join("");
    const benchCards = [];
    for (let i = 0; i < TeamBuilder.MAX_BENCH; i++) {
        benchCards.push(await rosterCard(current.bench[i], current.bench[i]?.slot || "BENCH"));
    }
    document.querySelector("#benchGrid").innerHTML = benchCards.join("");
    bindTeamActions();
}

function bindTeamActions() {
    document.querySelectorAll("[data-team-add], [data-team-replace]").forEach(btn => btn.addEventListener("click", () => openTeamEditor(btn.dataset.teamAdd || btn.dataset.teamReplace, "replace")));
    document.querySelectorAll("[data-team-season]").forEach(btn => btn.addEventListener("click", () => openSeasonChanger(btn.dataset.teamSeason)));
    document.querySelectorAll("[data-team-detail]").forEach(btn => btn.addEventListener("click", () => showRosterDetails(btn.dataset.teamDetail)));
    document.querySelectorAll("[data-team-remove]").forEach(btn => btn.addEventListener("click", () => {
        setTeam(TeamBuilder.remove(team(), btn.dataset.teamRemove));
        renderRoster();
    }));
    document.querySelectorAll("[data-team-move]").forEach(sel => sel.addEventListener("change", e => {
        if (!e.target.value) return;
        const result = TeamBuilder.move(team(), e.target.dataset.teamMove, e.target.value);
        if (result.error) showTeamMessage(result.error, true);
        else setTeam(result.team);
        renderRoster();
    }));
}

function showTeamMessage(message, error = false) {
    const box = document.querySelector("#teamBuilderMessage");
    if (!box) return;
    box.innerHTML = message ? `<div class="${error ? "error" : "notice"}">${text(message)}</div>` : "";
}

async function openTeamEditor(slot, mode = "replace") {
    state.teamEditor = { slot, player: null, profile: null, mode };
    const editor = document.querySelector("#teamEditor");
    editor.innerHTML = `
        <div class="panel control-stack">
            <div class="section-header"><div><div class="eyebrow">Add Player</div><h2>${slot === "BENCH" ? "Bench" : slot}</h2></div><button class="button secondary" id="closeTeamEditor">Close</button></div>
            <div class="field autocomplete"><label for="teamPlayerSearch">Historical player</label><input id="teamPlayerSearch" placeholder="Search player..." autocomplete="off" /><div class="suggestions" id="teamPlayerSuggestions"></div></div>
            <div class="field" id="teamSeasonSelectWrap"><label>Season</label><select disabled><option>Select player first</option></select></div>
            <button class="button" id="confirmTeamPlayer" disabled>Add to roster</button>
        </div>`;
    document.querySelector("#closeTeamEditor").addEventListener("click", () => { editor.innerHTML = ""; });
    document.querySelector("#teamPlayerSearch").addEventListener("input", e => searchHistoricalPlayers(e.target.value, "#teamPlayerSuggestions", async player => {
        state.teamEditor.player = player;
        document.querySelector("#teamPlayerSearch").value = player.playerName;
        const profile = await api(`/api/historical/players/${encodeURIComponent(player.playerId)}`);
        state.teamEditor.profile = profile;
        const selected = profile.seasons[profile.seasons.length - 1].season;
        document.querySelector("#teamSeasonSelectWrap").innerHTML = `<label for="teamSeasonSelect">Season</label><select id="teamSeasonSelect">${seasonOptions(profile, selected)}</select>`;
        document.querySelector("#confirmTeamPlayer").disabled = false;
    }));
    document.querySelector("#confirmTeamPlayer").addEventListener("click", async () => {
        const selectedSeason = document.querySelector("#teamSeasonSelect")?.value;
        const player = state.teamEditor.player;
        if (!player || !selectedSeason) return;
        const profile = await api(`/api/historical/players/${encodeURIComponent(player.playerId)}/seasons/${encodeURIComponent(selectedSeason)}`);
        const entry = TeamBuilder.entryFromSeason(profile, slot);
        const result = TeamBuilder.addOrReplace(team(), slot, entry);
        if (result.error) {
            showTeamMessage(result.error, true);
            return;
        }
        setTeam(result.team);
        editor.innerHTML = "";
        showTeamMessage(`${entry.playerName} ${entry.season} added.`);
        renderRoster();
    });
}

async function openSeasonChanger(slot) {
    const entry = TeamBuilder.rosterEntries(team()).find(e => e.slot === slot);
    if (!entry) return;
    const editor = document.querySelector("#teamEditor");
    const profile = await api(`/api/historical/players/${encodeURIComponent(entry.playerId)}`);
    editor.innerHTML = `
        <div class="panel control-stack">
            <div class="section-header"><div><div class="eyebrow">Change Season</div><h2>${text(entry.playerName)}</h2></div><button class="button secondary" id="closeTeamEditor">Close</button></div>
            <div class="field"><label for="teamSeasonChange">Season</label><select id="teamSeasonChange">${seasonOptions(profile, entry.season)}</select></div>
            <button class="button" id="confirmSeasonChange">Update season</button>
        </div>`;
    document.querySelector("#closeTeamEditor").addEventListener("click", () => { editor.innerHTML = ""; });
    document.querySelector("#confirmSeasonChange").addEventListener("click", async () => {
        const seasonProfile = await api(`/api/historical/players/${encodeURIComponent(entry.playerId)}/seasons/${encodeURIComponent(document.querySelector("#teamSeasonChange").value)}`);
        const result = TeamBuilder.changeSeason(team(), slot, seasonProfile);
        if (result.error) showTeamMessage(result.error, true);
        else {
            setTeam(result.team);
            editor.innerHTML = "";
            renderRoster();
        }
    });
}

async function showRosterDetails(slot) {
    const entry = TeamBuilder.rosterEntries(team()).find(e => e.slot === slot);
    if (!entry) return;
    const profile = await profileForEntry(entry);
    document.querySelector("#teamDetails").innerHTML = `<div class="panel">${seasonProfileHtml(profile)}</div>`;
}

async function importTeamJson(raw) {
    const parsed = TeamBuilder.deserialize(raw);
    const entries = TeamBuilder.rosterEntries(parsed);
    for (const entry of entries) {
        await api(`/api/historical/players/${encodeURIComponent(entry.playerId)}/seasons/${encodeURIComponent(entry.season)}`);
    }
    setTeam(parsed);
    await renderRoster();
}

async function renderTeamBuilder() {
    state.teamBuilder = TeamBuilder.load();
    app.innerHTML = `
        <section class="section">
            <div class="section-header">
                <div><div class="eyebrow">Team Builder</div><h2>Build an all-time roster.</h2><p class="muted">Select specific historical player-seasons. No ratings, chemistry, or simulation outputs are generated here.</p></div>
                <button class="button secondary" id="clearTeam">Clear team</button>
            </div>
            <div class="team-builder-layout">
                <div class="panel control-stack">
                    <div class="field"><label for="teamName">Team name</label><input id="teamName" value="${text(team().teamName)}" /></div>
                    <div class="button-row">
                        <button class="button secondary" id="exportTeam">Export Team JSON</button>
                        <button class="button secondary" id="importTeam">Import Team JSON</button>
                    </div>
                    <textarea id="teamJsonBox" rows="8" placeholder="Team JSON import/export" aria-label="Team JSON import export"></textarea>
                    <div id="teamBuilderMessage"></div>
                </div>
                <div>
                    <div id="teamSummary"></div>
                    <section class="section">
                        <div class="section-header"><div><h2>Starting five</h2><p class="muted">PG, SG, SF, PF, C slots guide roster shape without rigid position blocking.</p></div></div>
                        <div class="roster-grid" id="startersGrid"></div>
                    </section>
                    <section class="section">
                        <div class="section-header"><div><h2>Bench</h2><p class="muted">Up to seven additional player-seasons.</p></div><button class="button secondary" data-team-add="BENCH">Add bench player</button></div>
                        <div class="roster-grid" id="benchGrid"></div>
                    </section>
                    <div id="teamEditor"></div>
                    <div id="teamDetails"></div>
                </div>
            </div>
        </section>`;
    document.querySelector("#teamName").addEventListener("input", e => {
        const next = { ...team(), teamName: e.target.value || "My All-Time Team" };
        setTeam(next);
    });
    document.querySelector("#clearTeam").addEventListener("click", () => {
        setTeam(TeamBuilder.createEmptyTeam());
        state.teamProfiles.clear();
        renderTeamBuilder();
    });
    document.querySelector("#exportTeam").addEventListener("click", () => {
        document.querySelector("#teamJsonBox").value = JSON.stringify(TeamBuilder.exportTeam(team()), null, 2);
    });
    document.querySelector("#importTeam").addEventListener("click", async () => {
        try {
            await importTeamJson(document.querySelector("#teamJsonBox").value);
            showTeamMessage("Team imported.");
        } catch (err) {
            showTeamMessage(`Invalid imported roster: ${err.message}`, true);
        }
    });
    document.querySelector("[data-team-add='BENCH']").addEventListener("click", () => openTeamEditor("BENCH"));
    await renderRoster();
}

async function renderHistory() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Historical Decisions</div><h2>Decision Policy v1 explorer.</h2><p class="muted">Paginated results from held-out 2025-26 regular season artifacts.</p></div></div>
            <div class="panel control-stack">
                <div class="toolbar">
                    <input id="histPlayer" placeholder="Filter player name" style="max-width:260px" />
                    <select id="histDecision"><option value="">All recommendations</option><option>TAKE</option><option>REVIEW</option><option>DEFER</option></select>
                    <select id="histClutch"><option value="">All contexts</option><option value="true">Clutch only</option><option value="false">Non-clutch</option></select>
                    <button class="button secondary" id="histApply">Apply</button>
                </div>
                <div id="historyTable"></div>
            </div>
        </section>
        <section class="section two-col">
            ${panel("Decision distribution", bars(facts.regularSummary))}
            ${panel("Clutch distribution", bars(facts.clutchSummary))}
        </section>`;
    document.querySelector("#histApply").addEventListener("click", () => { state.historyPage = 1; loadHistory(); });
    await loadHistory();
}

async function loadHistory() {
    const target = document.querySelector("#historyTable");
    target.innerHTML = `<div class="muted">Loading decisions...</div>`;
    const q = new URLSearchParams({ page: state.historyPage, pageSize: 20 });
    const player = document.querySelector("#histPlayer")?.value;
    const rec = document.querySelector("#histDecision")?.value;
    const clutch = document.querySelector("#histClutch")?.value;
    if (player) q.set("player", player);
    if (rec) q.set("recommendation", rec);
    if (clutch) q.set("clutchOnly", clutch);
    try {
        const data = await api(`/api/analysis/decisions?${q}`);
        target.innerHTML = data.items.length ? `
            <div class="table-wrap"><table><thead><tr><th>Player</th><th>Team</th><th>Context</th><th>p_make</th><th>xPPS</th><th>Margin</th><th>Decision</th><th>Result</th></tr></thead><tbody>
                ${data.items.map(r => `<tr><td>${r.player_name}</td><td>${r.team_name || "—"}</td><td>${r.is_clutch ? "Clutch" : "Regular"}</td><td>${pct(r.p_make, 1)}</td><td>${fmt(r.shot_xpps, 2)}</td><td>${fmt(r.decision_margin, 2)}</td><td>${decisionBadge(r.decision)}</td><td>${r.points_scored} pts</td></tr>`).join("")}
            </tbody></table></div>
            <div class="toolbar" style="justify-content:space-between;margin-top:12px"><span class="small">Page ${data.page} · ${data.total.toLocaleString()} rows</span><div><button class="button secondary" id="prevPage">Previous</button> <button class="button secondary" id="nextPage">Next</button></div></div>` : `<div class="notice">No decisions match these filters.</div>`;
        document.querySelector("#prevPage")?.addEventListener("click", () => { state.historyPage = Math.max(1, state.historyPage - 1); loadHistory(); });
        document.querySelector("#nextPage")?.addEventListener("click", () => { state.historyPage += 1; loadHistory(); });
    } catch (err) {
        target.innerHTML = `<div class="error">${err.message}</div>`;
    }
}

function renderModel() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Model</div><h2>Calibrated shot quality, not black-box theater.</h2><p class="muted">Historical shooter profiles feed a HistGradientBoosting model; Decision Policy v1 compares expected shot value to a global replacement benchmark.</p></div></div>
            <div class="metrics-grid">${metric(compact(facts.shots), "canonical shots")}${metric(compact(facts.clutch), "clutch attempts")}${metric("2020-24", "training seasons")}${metric("2025-26", "held-out test")}</div>
        </section>
        <section class="section two-col">
            ${panel("Test performance", `<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>AUC</th><th>Log loss</th><th>Brier</th></tr></thead><tbody>${modelMetrics.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("")}</tbody></table></div>`)}
            ${panel("Calibration", `<svg viewBox="0 0 320 220" class="shot-map" role="img" aria-label="Calibration sketch"><line x1="35" y1="185" x2="285" y2="35" stroke="#73706a"/><polyline points="35,178 78,154 122,132 166,103 210,82 254,55" fill="none" stroke="#b7d7ff" stroke-width="3"/><text x="35" y="207" fill="#a5a29b" font-size="12">Predicted</text><text x="6" y="42" fill="#a5a29b" font-size="12" transform="rotate(-90 6 42)">Actual</text></svg><p class="small">Calibration is emphasized because the policy consumes probabilities as expected value.</p>`)}
        </section>
        <section class="section two-col">
            ${panel("Predictive features", `<div class="pill-row">${["shot distance", "action type", "vertical location", "historical zone FG%", "time remaining", "historical distance FG%", "player shooting profile", "score differential"].map(x => `<span class="pill">${x}</span>`).join("")}</div><p class="small" style="margin-top:14px">Labels are predictive, not causal.</p>`)}
            ${panel("Playoff limitation", `<p class="muted">The model remains discriminative in playoffs, but absolute probability calibration drifts on held-out playoff data. The UI does not recalibrate on playoff test results.</p>`)}
        </section>`;
}

function renderResearch() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Research</div><h2>What became production, and what did not.</h2></div></div>
            <div class="three-col">
                ${panel("Legacy Agent", `<p class="muted">Original epsilon-greedy Q-table over binned clutch state with a synthetic DON'T SHOOT baseline. Preserved as historical implementation.</p>`)}
                ${panel("Production v1", `<p class="muted">Calibrated shot quality, historical shooter profiles, global replacement xPPS, and TAKE / REVIEW / DEFER policy.</p>`)}
                ${panel("Possession Research", `<p class="muted">Possession-continuation modeling was explored but not promoted: PBP cannot observe true declined-shot counterfactuals, and the initial parser/target was not sufficiently defensible.</p>`)}
            </div>
        </section>`;
}

async function init() {
    navToggle.addEventListener("click", () => {
        nav.classList.toggle("open");
        navToggle.setAttribute("aria-expanded", nav.classList.contains("open"));
    });
    try {
        state.model = await api("/api/model");
        setStatus(true, "Production API online");
    } catch {
        setStatus(false, "Inference service unavailable");
    }
    window.addEventListener("hashchange", render);
    render();
}

function render() {
    setActiveNav();
    nav.classList.remove("open");
    const p = pageName();
    if (p === "lab") renderLab();
    else if (p === "players") renderPlayers();
    else if (p === "compare") renderCompare();
    else if (p === "team-builder") renderTeamBuilder();
    else if (p === "history") renderHistory();
    else if (p === "model") renderModel();
    else if (p === "research") renderResearch();
    else renderOverview();
    app.focus({ preventScroll: true });
}

init();
