const app = document.querySelector("#app");
const apiStatus = document.querySelector("#apiStatus");
const nav = document.querySelector("#mainNav");
const navToggle = document.querySelector("#navToggle");

const state = {
    model: null,
    selectedPlayer: null,
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
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
    return Number(n).toFixed(digits);
}

function pct(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
    return `${(Number(n) * 100).toFixed(digits)}%`;
}

function compact(n) {
    return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(n);
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
                <div class="eyebrow">Production shot decision intelligence</div>
                <h1>NBA Clutch Agent</h1>
                <p class="lede">Decision intelligence for NBA shot selection. The system combines a calibrated shot-quality model with a transparent replacement benchmark to return TAKE, REVIEW, or DEFER.</p>
                <div class="hero-actions">
                    <a class="button" href="#lab">Open Decision Lab</a>
                    <a class="button secondary" href="#model">View methodology</a>
                </div>
                <div class="metrics-grid">
                    ${metric(compact(facts.shots), "canonical shot attempts")}
                    ${metric(compact(facts.clutch), "clutch attempts")}
                    ${metric("6", "NBA seasons")}
                    ${metric("Decision v1", "production policy")}
                </div>
            </div>
            <div class="panel">
                <div class="eyebrow">System path</div>
                <div class="flow" style="margin-top:16px">
                    ${["Historical NBA data", "Shot Quality Model", "Expected Shot Value", "Decision Policy", "TAKE / REVIEW / DEFER"].map((x, i) => `<div class="flow-step"><span>0${i + 1}</span><strong>${x}</strong></div>`).join("")}
                </div>
                <div class="section">
                    <h3>2025-26 held-out regular season</h3>
                    ${bars(facts.regularSummary)}
                    <p class="small" style="margin-top:14px">Observed historical PPS is descriptive only. It is not agent-created value.</p>
                </div>
            </div>
        </section>
        <section class="section">
            <div class="section-header">
                <div><h2>Clutch analysis stays central.</h2><p class="muted">Clutch: final 5:00 of Q4 or overtime with score margin <= 5.</p></div>
                <a class="button secondary" href="#history">Explore history</a>
            </div>
            <div class="three-col">
                ${facts.clutchSummary.map(r => panel(decisionBadge(r.decision), `<div class="mini-panel"><div class="label">Share</div><div class="num">${pct(r.share)}</div></div><div class="mini-panel"><div class="label">Observed PPS</div><div class="num">${fmt(r.observed, 3)}</div></div>`)).join("")}
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
                <div><div class="eyebrow">Decision Lab</div><h2>Analyze a shot opportunity.</h2><p class="muted">Click the court, choose game context, and call the production model.</p></div>
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

async function renderPlayers() {
    app.innerHTML = `
        <section class="section">
            <div class="section-header"><div><div class="eyebrow">Players</div><h2>Search shooter profiles.</h2><p class="muted">Profiles are production artifacts derived from historical shooter features.</p></div></div>
            <div class="two-col">
                <div class="panel">
                    <div class="field autocomplete"><label for="playerSearch">Player search</label><input id="playerSearch" placeholder="Try Curry, Jokic, Booker..." /><div class="suggestions" id="playerSuggestions"></div></div>
                    <div id="playerDetail" style="margin-top:18px" class="small">Choose a player to inspect profile rows.</div>
                </div>
                <div class="panel"><h3>Profile fields</h3><p class="muted">The API resolves prior attempts, shrunk FG%, 2P/3P history, zone profile, distance profile, season history, previous-season profile, and cold-start state.</p><div id="playerBars"></div></div>
            </div>
        </section>`;
    document.querySelector("#playerSearch").addEventListener("input", e => searchPlayers(e.target.value, "#playerSuggestions"));
    document.querySelector("#playerSuggestions").addEventListener("click", async () => {
        setTimeout(async () => {
            if (!state.selectedPlayer) return;
            const detail = document.querySelector("#playerDetail");
            detail.innerHTML = `<div class="muted">Loading...</div>`;
            const data = await api(`/api/players/${state.selectedPlayer.player_id}/profile`);
            const p = data.profiles[0];
            detail.innerHTML = `<h3>${state.selectedPlayer.player_name}</h3><div class="metrics-grid">${metric(p.player_prior_fga, "prior FGA")}${metric(pct(p.player_prior_fg_pct_shrunk, 1), "shrunk FG%")}${metric(pct(p.player_prior_2p_pct_shrunk, 1), "shrunk 2P%")}${metric(pct(p.player_prior_3p_pct_shrunk, 1), "shrunk 3P%")}</div>`;
            document.querySelector("#playerBars").innerHTML = bars([
                { decision: "TAKE", share: p.player_prior_fg_pct_shrunk },
                { decision: "REVIEW", share: p.player_prior_2p_pct_shrunk },
                { decision: "DEFER", share: p.player_prior_3p_pct_shrunk },
            ]).replaceAll("TAKE", "FG").replaceAll("REVIEW", "2P").replaceAll("DEFER", "3P");
        }, 0);
    });
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
    else if (p === "history") renderHistory();
    else if (p === "model") renderModel();
    else if (p === "research") renderResearch();
    else renderOverview();
    app.focus({ preventScroll: true });
}

init();
