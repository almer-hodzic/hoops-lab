import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync("nba-clutch-agent/AiAgents.NbaClutchAgent.Web/wwwroot/teamBuilder.js", "utf8");
const storage = new Map();
const sandbox = {
  window: {
    localStorage: {
      setItem: (key, value) => storage.set(key, value),
      getItem: key => storage.get(key) ?? null,
    },
  },
  JSON,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
const TB = sandbox.window.TeamBuilder;

function profile(playerId, playerName, season, team, position) {
  return { identity: { playerId, playerName, season, team, position } };
}

const curry = TB.entryFromSeason(profile("br:curryst01", "Stephen Curry", "2015-16", "GSW", "PG"), "PG");
const jordan91 = TB.entryFromSeason(profile("br:jordami01", "Michael Jordan", "1990-91", "CHI", "SG"), "SG");
const jordan96 = TB.entryFromSeason(profile("br:jordami01", "Michael Jordan", "1995-96", "CHI", "SG"), "SF");
const lebron = TB.entryFromSeason(profile("br:jamesle01", "LeBron James", "2012-13", "MIA", "SF"), "SF");

let team = TB.createEmptyTeam("The Kings");
assert.equal(team.teamName, "The Kings");
assert.equal(TB.rosterSize(team), 0);

let result = TB.addOrReplace(team, "PG", curry);
assert.equal(result.error, null);
team = result.team;
assert.equal(team.starters.PG.playerId, "br:curryst01");

result = TB.addOrReplace(team, "SG", jordan91);
assert.equal(result.error, null);
team = result.team;
assert.equal(TB.rosterSize(team), 2);

result = TB.addOrReplace(team, "SF", jordan96);
assert.match(result.error, /already on the roster/);

result = TB.addOrReplace(team, "SF", lebron);
assert.equal(result.error, null);
team = result.team;

const changed = TB.changeSeason(team, "SF", profile("br:jamesle01", "LeBron James", "2015-16", "CLE", "SF"));
assert.equal(changed.error, null);
team = changed.team;
assert.equal(team.starters.SF.season, "2015-16");
assert.equal(team.starters.SF.playerId, "br:jamesle01");

const replacement = TB.addOrReplace(team, "SF", TB.entryFromSeason(profile("br:duncati01", "Tim Duncan", "2002-03", "SAS", "PF"), "SF"));
assert.equal(replacement.error, null);
team = replacement.team;
assert.equal(team.starters.SF.playerId, "br:duncati01");

team = TB.remove(team, "SG");
assert.equal(team.starters.SG, null);
assert.equal(TB.rosterSize(team), 2);

for (let i = 0; i < 10; i++) {
  result = TB.addOrReplace(team, "BENCH", TB.entryFromSeason(profile(`br:bench${i}`, `Bench ${i}`, "2000-01", "TOT", "G"), "BENCH"));
  if (!result.error) team = result.team;
}
assert.equal(team.bench.length, 7);
assert.match(result.error, /Bench is full|Maximum roster size/);

const fullTeam = TB.createEmptyTeam("Full");
let filled = fullTeam;
for (const slot of TB.STARTER_SLOTS) {
  filled = TB.addOrReplace(filled, slot, TB.entryFromSeason(profile(`br:${slot}`, slot, "1990-91", "TOT", slot), slot)).team;
}
for (let i = 0; i < 7; i++) {
  filled = TB.addOrReplace(filled, "BENCH", TB.entryFromSeason(profile(`br:f${i}`, `F ${i}`, "1990-91", "TOT", "G"), "BENCH")).team;
}
assert.equal(TB.rosterSize(filled), 12);
assert.match(TB.addOrReplace(filled, "BENCH", TB.entryFromSeason(profile("br:extra", "Extra", "1990-91", "TOT", "G"), "BENCH")).error, /Maximum roster size|Bench is full/);

TB.save(team, sandbox.window.localStorage);
const loaded = TB.load(sandbox.window.localStorage);
assert.equal(loaded.teamName, "The Kings");
assert.equal(loaded.starters.PG.playerId, "br:curryst01");

assert.equal(TB.deserialize("{not-json").teamName, "My All-Time Team");

const exported = TB.exportTeam(team);
assert.equal(exported.name, "The Kings");
assert.equal(exported.starters[0].playerId, "br:curryst01");
assert.equal(Object.hasOwn(exported.starters[0], "pointsPer75"), false);
const imported = TB.deserialize(JSON.stringify(exported));
assert.equal(imported.teamName, "The Kings");
assert.equal(imported.starters.PG.playerId, "br:curryst01");

console.log(JSON.stringify({ status: "pass" }, null, 2));
