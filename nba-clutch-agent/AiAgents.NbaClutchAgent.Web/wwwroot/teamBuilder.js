(function () {
    const VERSION = 1;
    const STORAGE_KEY = "hoopslab.teamBuilder.v1";
    const STARTER_SLOTS = ["PG", "SG", "SF", "PF", "C"];
    const MAX_BENCH = 7;
    const MAX_ROSTER = 12;

    function createEmptyTeam(name = "My All-Time Team") {
        return {
            schemaVersion: VERSION,
            teamName: name,
            starters: Object.fromEntries(STARTER_SLOTS.map(slot => [slot, null])),
            bench: [],
        };
    }

    function rosterEntries(team) {
        return [
            ...STARTER_SLOTS.map(slot => team.starters?.[slot]).filter(Boolean),
            ...(team.bench || []),
        ];
    }

    function rosterSize(team) {
        return rosterEntries(team).length;
    }

    function normalizeEntry(entry, slot) {
        if (!entry || !entry.playerId || !entry.season) return null;
        return {
            slot,
            playerId: String(entry.playerId),
            playerName: String(entry.playerName || ""),
            season: String(entry.season),
            team: String(entry.team || ""),
            position: entry.position ? String(entry.position) : null,
        };
    }

    function hasDuplicatePlayer(team, playerId, exceptSlot = null) {
        return rosterEntries(team).some(entry => entry.playerId === playerId && entry.slot !== exceptSlot);
    }

    function validateEntry(team, entry, targetSlot, options = {}) {
        if (!entry || !entry.playerId || !entry.season) return { ok: false, error: "Select a real historical player-season." };
        if (!targetSlot) return { ok: false, error: "Choose a roster slot." };
        if (!STARTER_SLOTS.includes(targetSlot) && targetSlot !== "BENCH") return { ok: false, error: "Invalid roster slot." };
        const exceptSlot = options.replace ? targetSlot : null;
        if (hasDuplicatePlayer(team, entry.playerId, exceptSlot)) return { ok: false, error: "This player is already on the roster. Choose one historical version." };
        const sizeAfter = rosterSize(team) + (options.replace && findEntry(team, targetSlot) ? 0 : 1);
        if (sizeAfter > MAX_ROSTER) return { ok: false, error: "Maximum roster size is 12 players." };
        if (targetSlot === "BENCH" && !options.replace && (team.bench || []).length >= MAX_BENCH) return { ok: false, error: "Bench is full." };
        return { ok: true, error: null };
    }

    function cloneTeam(team) {
        return JSON.parse(JSON.stringify(team || createEmptyTeam()));
    }

    function addOrReplace(team, targetSlot, entry) {
        const next = cloneTeam(team);
        const replace = targetSlot !== "BENCH";
        const validation = validateEntry(next, entry, targetSlot, { replace });
        if (!validation.ok) return { team: next, error: validation.error };
        if (targetSlot === "BENCH") {
            next.bench.push(normalizeEntry(entry, `BENCH${next.bench.length + 1}`));
        } else {
            next.starters[targetSlot] = normalizeEntry(entry, targetSlot);
        }
        resequenceBench(next);
        return { team: next, error: null };
    }

    function remove(team, slot) {
        const next = cloneTeam(team);
        if (STARTER_SLOTS.includes(slot)) {
            next.starters[slot] = null;
        } else if (slot?.startsWith("BENCH")) {
            const index = Number(slot.replace("BENCH", "")) - 1;
            if (index >= 0) next.bench.splice(index, 1);
        }
        resequenceBench(next);
        return next;
    }

    function findEntry(team, slot) {
        if (STARTER_SLOTS.includes(slot)) return team.starters?.[slot] || null;
        if (slot?.startsWith("BENCH")) return team.bench?.[Number(slot.replace("BENCH", "")) - 1] || null;
        return null;
    }

    function move(team, fromSlot, toSlot) {
        const next = cloneTeam(team);
        const entry = findEntry(next, fromSlot);
        if (!entry || fromSlot === toSlot) return { team: next, error: null };
        if (toSlot === "BENCH" && STARTER_SLOTS.includes(fromSlot) && (next.bench || []).length >= MAX_BENCH) {
            return { team: next, error: "Bench is full." };
        }
        const target = findEntry(next, toSlot);
        if (STARTER_SLOTS.includes(fromSlot)) next.starters[fromSlot] = null;
        else next.bench.splice(Number(fromSlot.replace("BENCH", "")) - 1, 1);
        if (STARTER_SLOTS.includes(toSlot)) {
            if (target) {
                if (STARTER_SLOTS.includes(fromSlot)) next.starters[fromSlot] = normalizeEntry(target, fromSlot);
                else next.bench.push(normalizeEntry(target, `BENCH${next.bench.length + 1}`));
            }
            next.starters[toSlot] = normalizeEntry(entry, toSlot);
        } else {
            next.bench.push(normalizeEntry(entry, `BENCH${next.bench.length + 1}`));
        }
        resequenceBench(next);
        return { team: next, error: null };
    }

    function changeSeason(team, slot, seasonProfile) {
        const existing = findEntry(team, slot);
        if (!existing || !seasonProfile?.identity) return { team: cloneTeam(team), error: "Invalid PlayerSeason." };
        const nextEntry = entryFromSeason(seasonProfile, slot);
        const next = cloneTeam(team);
        if (STARTER_SLOTS.includes(slot)) next.starters[slot] = nextEntry;
        else next.bench[Number(slot.replace("BENCH", "")) - 1] = nextEntry;
        resequenceBench(next);
        return { team: next, error: null };
    }

    function entryFromSeason(profile, slot) {
        const identity = profile.identity || profile;
        return normalizeEntry({
            playerId: identity.playerId,
            playerName: identity.playerName,
            season: identity.season,
            team: identity.team,
            position: identity.position,
        }, slot);
    }

    function resequenceBench(team) {
        team.bench = (team.bench || []).filter(Boolean).slice(0, MAX_BENCH).map((entry, index) => normalizeEntry(entry, `BENCH${index + 1}`));
    }

    function sanitizeTeam(value) {
        if (!value || value.schemaVersion !== VERSION) return createEmptyTeam();
        const team = createEmptyTeam(String(value.teamName || value.name || "My All-Time Team"));
        const rawStarters = Array.isArray(value.starters)
            ? Object.fromEntries(value.starters.map(entry => [entry.slot, entry]))
            : value.starters;
        for (const slot of STARTER_SLOTS) {
            team.starters[slot] = normalizeEntry(rawStarters?.[slot], slot);
        }
        team.bench = (value.bench || []).slice(0, MAX_BENCH).map((entry, index) => normalizeEntry(entry, `BENCH${index + 1}`)).filter(Boolean);
        const seen = new Set();
        for (const slot of STARTER_SLOTS) {
            const entry = team.starters[slot];
            if (entry && seen.has(entry.playerId)) team.starters[slot] = null;
            if (entry) seen.add(entry.playerId);
        }
        team.bench = team.bench.filter(entry => {
            if (seen.has(entry.playerId)) return false;
            seen.add(entry.playerId);
            return true;
        });
        return team;
    }

    function serialize(team) {
        return JSON.stringify(sanitizeTeam(team), null, 2);
    }

    function deserialize(raw) {
        try {
            return sanitizeTeam(JSON.parse(raw));
        } catch {
            return createEmptyTeam();
        }
    }

    function save(team, storage = window.localStorage) {
        storage.setItem(STORAGE_KEY, serialize(team));
    }

    function load(storage = window.localStorage) {
        const raw = storage.getItem(STORAGE_KEY);
        return raw ? deserialize(raw) : createEmptyTeam();
    }

    function exportTeam(team) {
        const clean = sanitizeTeam(team);
        return {
            name: clean.teamName,
            schemaVersion: VERSION,
            starters: STARTER_SLOTS.map(slot => clean.starters[slot]).filter(Boolean).map(entry => ({
                slot: entry.slot,
                playerId: entry.playerId,
                season: entry.season,
            })),
            bench: clean.bench.map(entry => ({
                slot: entry.slot,
                playerId: entry.playerId,
                season: entry.season,
            })),
        };
    }

    window.TeamBuilder = {
        VERSION,
        STORAGE_KEY,
        STARTER_SLOTS,
        MAX_BENCH,
        MAX_ROSTER,
        createEmptyTeam,
        rosterEntries,
        rosterSize,
        validateEntry,
        addOrReplace,
        remove,
        move,
        changeSeason,
        entryFromSeason,
        serialize,
        deserialize,
        save,
        load,
        exportTeam,
    };
})();
