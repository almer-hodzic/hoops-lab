"""Frozen V1 roster/game inference. Imports only approved pure research kernels."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

import defensive_continuity as defense
import finalize_game_calibration as game
import improve_projection as team

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data/simulation/production_v1"
SLOTS = ("PG", "SG", "SF", "PF", "C")
DISCLAIMER = "Uncertainty-aware statistical model, not objective historical truth. Outcome intervals do not exhaust custom-roster uncertainty."


def text_fingerprint(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class SimulationError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class FrozenPipeline:
    """Read-only imputer/scaler/linear predictor exported from fitted sklearn state."""
    def __init__(self, state):
        self.state = {key: np.asarray(value, dtype=float) for key, value in state.items()}

    def predict(self, frame):
        state = self.state
        values = np.asarray(frame, dtype=float)
        values = np.where(np.isnan(values), state["impute"], values)
        return ((values - state["center"]) / state["scale"]) @ state["coefficients"] + state["intercept"]


def candidate(state):
    return (FrozenPipeline(state["pipeline"]) if state["pipeline"] else None, state["columns"])


def draw_scores(frame, home, sampler, overtime, count, seed):
    """Approved simulate() sampling path, adapted only to retain score/OT draws.

    Paired with exact same-seed parity against finalize_game_calibration.simulate.
    Regulation and joint resampling are shared, not reimplemented.
    """
    rng = np.random.default_rng(seed)
    means = game.means(frame, home)
    hp, ap, raw_margin, _ = game.regulation(frame, means, sampler, count, rng)
    close = (abs(hp-ap) >= 1) & (abs(hp-ap) <= 3)
    pin = close & (rng.random(hp.shape) < overtime["extra"])
    hp[pin] = np.rint((hp[pin]+ap[pin])/2).astype(int)
    ap[pin] = hp[pin]
    thin = (hp == ap) & (raw_margin != 0) & (rng.random(hp.shape) < overtime["thinning"])
    hp[thin & (raw_margin > 0)] += 1
    ap[thin & (raw_margin < 0)] += 1
    regulation_a, regulation_b = hp.copy(), ap.copy()
    tied = hp == ap
    periods = np.zeros(hp.shape, dtype=int)
    for _ in range(20):
        if not tied.any():
            break
        sample = overtime["samples"][rng.integers(0, len(overtime["samples"]), size=int(tied.sum()))]
        scale = np.repeat((means["pace"]*frame.league_ortg.to_numpy()/100*5/48)[:, None], count, axis=1)[tied]
        hmean = np.repeat((means["home"]*5/48)[:, None], count, axis=1)[tied]
        amean = np.repeat((means["away"]*5/48)[:, None], count, axis=1)[tied]
        hp[tied] += np.maximum(0, np.rint(hmean+sample[:, 0]*scale)).astype(int)
        ap[tied] += np.maximum(0, np.rint(amean+sample[:, 1]*scale)).astype(int)
        periods[tied] += 1
        tied = hp == ap
    if tied.any():
        raise RuntimeError("Overtime failed to resolve; final ties are never returned")
    return hp[0], ap[0], regulation_a[0], regulation_b[0], periods[0], means


class ProductionSimulator:
    def __init__(self, artifact_dir=ARTIFACTS, historical_dir=None):
        historical_dir = historical_dir or ROOT / "data/historical/processed"
        self.config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        self.artifact_sha256 = text_fingerprint(artifact_dir / "config.json")
        for name, expected in self.config["files"].items():
            if text_fingerprint(artifact_dir / name) != expected:
                raise RuntimeError(f"Production artifact checksum mismatch: {name}")
        for name, expected in self.config["canonical_hashes"].items():
            if text_fingerprint(historical_dir / name) != expected:
                raise RuntimeError(f"Canonical data version mismatch: {name}")
        for name, expected in self.config["kernel_hashes"].items():
            module, function = name.rsplit(".", 1)
            source = inspect.getsource(getattr(importlib.import_module(module), function))
            if hashlib.sha256(source.encode()).hexdigest() != expected:
                raise RuntimeError(f"Approved numerical kernel changed: {name}")
        self.models = {key: candidate(value) for key, value in self.config["models"].items()}
        self.defensive_model = dict(candidate=candidate(self.config["defense"]), tail=None)
        self.sampler = dict(self.config["sampler"])
        self.sampler["covariance"] = np.array(self.sampler["covariance"])
        self.sampler["samples"] = np.loadtxt(artifact_dir / "joint_residuals.csv", delimiter=",", skiprows=1)
        self.overtime = dict(self.config["overtime"])
        self.overtime["samples"] = np.loadtxt(artifact_dir / "overtime_residuals.csv", delimiter=",", skiprows=1)
        players = pd.read_csv(historical_dir / "player_seasons.csv")
        supplemental = pd.read_csv(artifact_dir / "defensive_profiles.csv")
        profiles = team.previous.add_derived_player_features(players).merge(
            supplemental, on=["player_id", "season_end_year", "team"], how="left", validate="one_to_one")
        self.profiles = profiles.set_index(["player_id", "season"], drop=False)
        if not self.profiles.index.is_unique:
            raise RuntimeError("Canonical PlayerSeason identities are not unique")
        self.player_ids = set(players.player_id)
        self.histories = {pid: rows.sort_values("season_end_year") for pid, rows in profiles.groupby("player_id")}
        leagues = pd.read_csv(historical_dir / "league_seasons.csv")
        self.leagues = {row.season: row for row in leagues.itertuples() if row.season_end_year >= 1980}
        teams = pd.read_csv(historical_dir / "team_seasons.csv")
        self.schedules = teams.assign(games=teams.wins+teams.losses).groupby("season_end_year").games.max().to_dict()
        self.league_years = leagues.set_index("season_end_year").league_points_per_75.to_dict()
        # Same minutes-weighted traded-player context/origin mapping as final_game_engine.
        splits = pd.read_csv(historical_dir / "player_team_splits.csv")
        members = team.previous.roster_rows(players, splits)
        environment = teams[["season_end_year", "team_id", "def_rtg"]].copy()
        environment["relative_defense"] = environment.def_rtg-environment.season_end_year.map(self.league_years)/75*100
        members = members.merge(environment[["season_end_year", "team_id", "relative_defense"]],
                                on=["season_end_year", "team_id"], validate="many_to_one")
        members["weighted"] = members.relative_defense*members.minutes
        totals = members.groupby(["season_end_year", "player_id"])[["weighted", "minutes"]].sum()
        self.environments = totals.weighted.div(totals.minutes).to_dict()
        self.origins = members.sort_values("minutes", ascending=False).drop_duplicates(["season_end_year", "player_id"]).set_index(
            ["season_end_year", "player_id"]).franchise_id.to_dict()

    def info(self):
        return dict(modelVersion=self.config["modelVersion"], schemaVersion=1,
                    supportedTargetSeasons=sorted(self.leagues), venues=["neutral", "teamA_home", "teamB_home"],
                    **self.config["defaults"], starterSlots=list(SLOTS), ironmanPlayers=5,
                    calibrationEndSeason="2018-19", artifactSha256=self.artifact_sha256,
                    uncertaintyDisclaimer=DISCLAIMER)

    def validate(self, request):
        if not isinstance(request, dict):
            raise SimulationError("Request must be a JSON object")
        allowed = {"teamA", "teamB", "targetSeason", "venue", "simulationCount", "seed", "ironmanMode"}
        if set(request)-allowed:
            raise SimulationError("Unknown request fields: " + ", ".join(sorted(set(request)-allowed)))
        season = request.get("targetSeason")
        if not isinstance(season, str) or season not in self.leagues:
            raise SimulationError("Unsupported targetSeason")
        venue = request.get("venue", "neutral")
        if venue not in ("neutral", "teamA_home", "teamB_home"):
            raise SimulationError("Invalid venue")
        count = request.get("simulationCount", 10000)
        if type(count) is not int or not 1000 <= count <= 100000:
            raise SimulationError("simulationCount must be an integer from 1000 through 100000")
        seed = request.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**53-1):
            raise SimulationError("seed must be an integer from 0 through 9007199254740991, or null")
        ironman = request.get("ironmanMode", False)
        if type(ironman) is not bool:
            raise SimulationError("ironmanMode must be a boolean")
        rosters = [self.validate_team(request.get(key), ironman, key) for key in ["teamA", "teamB"]]
        return season, venue, count, seed, ironman, rosters

    def validate_team(self, value, ironman, label):
        if not isinstance(value, dict) or set(value)-{"name", "teamName", "schemaVersion", "starters", "bench"}:
            raise SimulationError(f"{label}: invalid team object")
        if "schemaVersion" in value and (type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1):
            raise SimulationError(f"{label}: unsupported Team Builder schemaVersion")
        name = value.get("name", value.get("teamName", label))
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise SimulationError(f"{label}: name must contain 1-100 characters")
        starters = value.get("starters")
        if isinstance(starters, dict):
            if set(starters) != set(SLOTS):
                raise SimulationError(f"{label}: starter keys must be PG/SG/SF/PF/C")
            entries = []
            for slot in SLOTS:
                row = starters[slot]
                if not isinstance(row, dict) or row.get("slot", slot) != slot:
                    raise SimulationError(f"{label}: malformed starter slot")
                entries.append(dict(row, slot=slot))
        elif isinstance(starters, list):
            entries = starters.copy()
            if len(entries) != 5 or any(not isinstance(e, dict) for e in entries):
                raise SimulationError(f"{label}: exactly five starters required")
            if any(not isinstance(e.get("slot"), str) for e in entries) or sorted(e["slot"] for e in entries) != sorted(SLOTS):
                raise SimulationError(f"{label}: unique PG/SG/SF/PF/C starter slots required")
            entries.sort(key=lambda e: SLOTS.index(e["slot"]))
        else:
            raise SimulationError(f"{label}: starters must name exactly five slots")
        bench = value.get("bench", [])
        if not isinstance(bench, list) or any(not isinstance(e, dict) for e in bench):
            raise SimulationError(f"{label}: bench must be an array of player entries")
        for i, row in enumerate(bench):
            if row.get("slot", f"BENCH{i+1}") not in ("BENCH", f"BENCH{i+1}"):
                raise SimulationError(f"{label}: malformed bench slot")
            entries.append(dict(row, slot=f"BENCH{i+1}"))
        if (ironman and len(entries) != 5) or (not ironman and not 8 <= len(entries) <= 12):
            raise SimulationError(f"{label}: ironman requires exactly 5 players; normal mode requires 8-12")
        identities = set()
        for row in entries:
            if set(row)-{"slot", "playerId", "season", "playerName", "team", "position"}:
                raise SimulationError(f"{label}: caller-supplied statistical/unknown fields are not allowed")
            pid, season = row.get("playerId"), row.get("season")
            if not isinstance(pid, str) or pid not in self.player_ids:
                raise SimulationError(f"{label}: unknown playerId")
            if not isinstance(season, str) or (pid, season) not in self.profiles.index:
                raise SimulationError(f"{label}: unavailable player season")
            if pid in identities:
                raise SimulationError(f"{label}: duplicate canonical player")
            identities.add(pid)
        return name.strip(), entries

    def project(self, roster, target_season, ironman):
        name, entries = roster
        records, missing, contexts, origins = [], [], [], []
        for entry in entries:
            pid, season = entry["playerId"], entry["season"]
            raw = self.profiles.loc[(pid, season)].copy()
            verified = pd.notna(raw.source_minutes) and abs(raw.source_minutes-raw.minutes) <= 1.
            evidence = [pd.notna(raw.get(field)) for field in defense.SOURCE_FIELDS]
            if not verified:
                for field in team.previous.DEFENSIVE_FIELDS.values():
                    raw[field] = np.nan
                evidence[0] = False
            missing.append(1-sum(evidence)/len(evidence))
            defaults = self.config["role_defaults"].get(str(raw.position), self.config["profile_defaults"])
            for field in team.previous.PROFILE_FIELDS:
                if pd.isna(raw[field]):
                    raw[field] = defaults[field]
            for field in ["team_pace", "league_pace"]:
                if pd.isna(raw[field]):
                    raw[field] = self.config["profile_defaults"][field]
            year = int(raw.season_end_year)
            history = self.histories[pid]
            history = history[(history.season_end_year <= year) & (history.season_end_year >= year-2) & (history.minutes > 0)].iloc[::-1]
            history_mpg = float(np.average(history.minutes_per_game, weights=.5**np.arange(len(history)))) if len(history) else 8.
            raw["prior_mpg"] = float(raw.minutes_per_game) if pd.notna(raw.minutes_per_game) else 8.
            raw["history_mpg"] = history_mpg
            raw["prior_availability"] = min(float(raw.games)/self.schedules[year], 1.) if raw.games > 0 else .5
            raw["prior_start_share"] = float(raw.games_started/raw.games) if raw.games > 0 and pd.notna(raw.games_started) else .5
            raw["projected_age"] = float(raw.age) if pd.notna(raw.age) else 26.
            raw["known"], raw["returner"], raw["profile_lag"] = 1., 0., 1.
            # Context dispersion is uncertainty evidence, never inherited team defense.
            env = self.environments.get((year, pid), 0.)
            raw["prior_defensive_environment"] = env
            contexts.append(env)
            origins.append((year, str(self.origins.get((year, pid), "unknown:"+pid))))
            raw["player_id"] = pid
            records.append(raw.to_dict())
        prior = pd.DataFrame(records)
        prior["prior_rank"] = prior.prior_mpg.rank(method="first", ascending=False)
        prior["roster_depth"] = len(prior)
        minutes = team.projected_rotation(prior, self.config["rotation"], self.config["rookie_priors"],
                                          mode="ironman" if ironman else "full", slots=[e["slot"] for e in entries])
        league = self.leagues[target_season]
        features = team.aggregate_case(dict(prior=prior, context=league, year=league.season_end_year,
            season=target_season, team_id="custom", team_name=name, actual={}), minutes,
            self.config["usage"], self.config["role_usage_means"])
        weights = minutes/240.
        features.update(defense.continuity(minutes, np.zeros(len(prior), bool), 0., synthetic=True))
        mean = float(weights @ contexts)
        origin_weights = pd.Series(weights).groupby(pd.Series(origins).map(str)).sum()
        features.update(defensive_missing_share=float(weights @ missing),
                        source_context_sd=float(np.sqrt(weights @ np.square(np.asarray(contexts)-mean))),
                        source_diversity=float(1-np.square(origin_weights).sum()))
        frame = pd.DataFrame([features])
        strengths = {target: float(team.previous.predict(frame, target, model)[0]) for target, model in self.models.items()}
        strengths["drtg"] = float(defense.predict(frame, self.defensive_model)[0])
        extrapolation = team.extrapolation(features, self.config["domain"])
        uncertainty = defense.uncertainty(features, self.config["defensive_domain"], self.config["uncertainty_thresholds"])
        reasons = [key for key, flagged in uncertainty["high_flags"].items() if flagged]
        if extrapolation["risk"] == "high":
            reasons.append("team_feature_extrapolation")
            uncertainty["category"] = "HIGH"
        if ironman:
            reasons.append("ironman_nonstandard_hypothetical_48_minutes_no_fatigue_model")
            uncertainty["category"] = "HIGH"
        effective, shares = team.usage(prior, minutes, self.config["usage"], self.config["role_usage_means"])
        rotation = [dict(playerId=r.player_id, playerName=r.player_name, season=r.season, slot=entries[i]["slot"],
                         minutes=float(minutes[i]), normalizedUsagePct=float(effective[i]), possessionShare=float(shares[i]))
                    for i, r in enumerate(prior.itertuples())]
        diagnostics = {key: features[key] for key in ["usage_weighted_relative_ts", "usage_weighted_3par",
            "usage_weighted_ftr", "w_assist_pct", "w_turnover_pct", "w_offensive_rebounds_per_75",
            "w_defensive_rebounds_per_75", "w_defensive_box_plus_minus", "usage_concentration", "usage_compression",
            "rotation_hhi", "defensive_missing_share", "returning_minutes_share", "retained_defensive_context",
            "source_context_sd", "source_diversity"]}
        output = dict(name=name, projectedORtg=strengths["ortg"], projectedDRtg=strengths["drtg"], projectedPace=strengths["pace"],
                      uncertainty=uncertainty["category"], uncertaintyReasons=reasons, rotation=rotation,
                      projectedMinutes=round(float(minutes.sum()), 10), rotationDepth=int((minutes >= 8).sum()),
                      extrapolation={key: value for key, value in extrapolation.items() if key != "features"},
                      defensiveCoverage={key: value for key, value in uncertainty.items() if key != "coverage"},
                      diagnostics=diagnostics)
        return output, features

    def game_inputs(self, a, b, season, venue):
        league = self.leagues[season]
        home_a = venue != "teamB_home"
        h, v = (a, b) if home_a else (b, a)
        frame = pd.DataFrame([dict(league_pace=league.league_pace, league_ortg=league.league_points_per_75/75*100,
            home_ortg=h["projectedORtg"], home_drtg=h["projectedDRtg"], home_pace=h["projectedPace"],
            away_ortg=v["projectedORtg"], away_drtg=v["projectedDRtg"], away_pace=v["projectedPace"])])
        home = self.config["home"] if venue != "neutral" else dict(common_offset=0., home_margin_per100=0.)
        return frame, home, home_a

    def simulate(self, request):
        season, venue, count, seed, ironman, rosters = self.validate(request)
        a, _ = self.project(rosters[0], season, ironman)
        b, _ = self.project(rosters[1], season, ironman)
        frame, home, home_a = self.game_inputs(a, b, season, venue)
        actual_seed = seed if seed is not None else secrets.randbits(53)
        ha, hb, ra, rb, periods, expected = draw_scores(frame, home, self.sampler, self.overtime, count, actual_seed)
        sa, sb = (ha, hb) if home_a else (hb, ha)
        reg_a, reg_b = (ra, rb) if home_a else (rb, ra)
        ea, eb = (expected["home"][0], expected["away"][0]) if home_a else (expected["away"][0], expected["home"][0])
        eff_a, eff_b = (expected["home_eff"][0], expected["away_eff"][0]) if home_a else (expected["away_eff"][0], expected["home_eff"][0])
        pa = float(np.mean(sa > sb))
        a.update(expectedPoints=float(ea), expectedEfficiency=float(eff_a), winProbability=pa)
        b.update(expectedPoints=float(eb), expectedEfficiency=float(eff_b), winProbability=1-pa)
        score_intervals, margin_intervals, regulation_intervals = {}, {}, {}
        for level in self.config["intervals"]:
            q = [(1-level)/2, (1+level)/2]
            key = str(round(level*100))
            score_intervals[key] = dict(teamA=np.quantile(sa, q).tolist(), teamB=np.quantile(sb, q).tolist())
            margin_intervals[key] = np.quantile(sa-sb, q).tolist()
            regulation_intervals[key] = np.quantile(reg_a-reg_b, q).tolist()
        # A compact exact empirical distribution, rather than returning 100,000 draws.
        def histogram(values):
            unique, frequencies = np.unique(values, return_counts=True)
            return [dict(value=int(v), count=int(n)) for v, n in zip(unique, frequencies)]
        league = self.leagues[season]
        return dict(modelVersion=self.config["modelVersion"], targetSeason=season, venue=venue,
            simulationCount=count, seed=actual_seed, ironmanMode=ironman, teamA=a, teamB=b,
            leagueEnvironment=dict(ORtg=float(frame.league_ortg.iloc[0]), pace=float(league.league_pace), trueShootingPct=float(league.league_ts_pct)),
            game=dict(expectedPace=float(expected["pace"][0]), expectedScore=dict(teamA=float(ea), teamB=float(eb)),
                expectedMargin=float(ea-eb), expectedTotal=float(ea+eb),
                simulatedMeanScore=dict(teamA=float(sa.mean()), teamB=float(sb.mean())),
                medianScore=dict(teamA=float(np.median(sa)), teamB=float(np.median(sb))),
                scoreIntervals=score_intervals, marginIntervals=margin_intervals, regulationMarginIntervals=regulation_intervals,
                scoreDistributions=dict(teamA=histogram(sa), teamB=histogram(sb), margin=histogram(sa-sb)),
                overtimeProbability=float(np.mean(periods > 0)), overtimePeriodDistribution=histogram(periods),
                finalTies=int(np.sum(sa == sb)), sampleOutcome=dict(teamA=int(sa[0]), teamB=int(sb[0]),
                    regulationTeamA=int(reg_a[0]), regulationTeamB=int(reg_b[0]), overtimePeriods=int(periods[0]))),
            metadata=dict(schemaVersion=1, researchCommit=self.config["researchCommit"], calibrationEndSeason="2018-19",
                artifactSha256=self.artifact_sha256, numpyVersion=np.__version__,
                inputMode="selected_historical_versions_custom_roster", sampler="D_bootstrap8_roster_engine",
                homeModel="trailing5_frozen_2019", overtimeModel="exponential5_frozen_2019",
                probabilityMonteCarloSE=float(np.sqrt(pa*(1-pa)/count)), disclaimer=DISCLAIMER,
                warnings=["Custom roster has no inherited franchise continuity", "Calibration ends 2018-19; defense and central interval coverage remain imperfect",
                          "Neutral and mixed-era rosters are extrapolations; no fatigue, injuries, coaching, or lineup interactions modeled"] +
                         (["IRONMAN: nonstandard hypothetical basketball; five players at 48 minutes each"] if ironman else [])))
