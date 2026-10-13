# NBA Clutch Agent

## Predikcija kvaliteta NBA šuta i podrška odlučivanju u clutch situacijama

## 1. O projektu i zašto je drugačiji

NBA Clutch Agent je akademski ML projekat koji predviđa vjerovatnoću pogođenog NBA šuta prije nego što je šut izveden, a zatim tu vjerovatnoću pretvara u očekivane poene po šutu. Projekat nije samo preuzeti CSV i jednostavni klasifikator: uključuje obradu NBA Shot Detail i PlayByPlay podataka, konstrukciju canonical dataseta, leakage-safe historijske profile šutera, kalibrirani model vjerovatnoće, decision policy i web aplikaciju za demonstraciju.

Produkcijski dio projekta koristi moderne NBA sezone 2020-21 do 2025-26. Eksperimenti sa defanzivnim tracking kontekstom iz 2014/15 tretiraju se kao odvojena istraživačka ekstenzija i ne utiču na produkcijsku preporuku.

## 2. Izvor i nastanak dataseta

Primarni dataset je `data/processed/shots_enriched.parquet`. Nastao je iz NBA Shot Detail arhiva i PlayByPlay V3 podataka. Shot Detail daje lokaciju, tip i rezultat šuta, dok PlayByPlay omogućava provjeru i rekonstrukciju pre-shot rezultata.

Dataset pokriva sezone 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26, regularnu sezonu i playoff. Ukupno sadrži 1,368,990 pokušaja šuta, 7,737 utakmica i 1,086 jedinstvenih šutera. Regularna sezona ima 1,282,307 redova, playoff 86,683, a clutch subset 46,813 redova.

Clutch je definisan kao četvrta četvrtina ili produžetak, 5:00 ili manje do kraja perioda, i apsolutna pre-shot razlika u rezultatu manja ili jednaka 5. Matching kvaliteta i audit nalaze se u `reports/data-audit/`.

## 3. Struktura podataka i ciljna varijabla

Ciljna varijabla je `shot_made`. U cijelom canonical datasetu pogođeno je 641,440 šuteva, a promašeno 727,550; ukupni FG% je 46.85%. Model koristi pre-shot geometriju, kontekst utakmice, kategorije šuta i historijski profil šutera.

2PT pokušaja ima 818,855, a 3PT pokušaja 550,135. Relevantne kolone i razlozi za uključivanje/isključivanje opisani su u `submission/data_dictionary.md`.

## 4. Analiza podataka (EDA)

Generisani grafici u `submission/report_assets/` prikazuju distribuciju targeta, distance šuta, uspješnost po zoni, clutch/non-clutch razlike, distribuciju predikcija i decision policy analizu. EDA potvrđuje očekivane košarkaške obrasce: bliski šutevi imaju viši procenat realizacije, 3PT šutevi imaju niži FG% ali veću vrijednost po pogotku, a clutch subset je manji i zahtijeva opreznu interpretaciju.

## 5. Priprema podataka

Shot i PBP podaci su spojeni da bi se dobio pre-shot kontekst, uključujući rezultat prije šuta. Redovi su hronološki uređeni, a split nije nasumičan: trening koristi regularne sezone 2020-21 do 2023-24, validacija 2024-25, a finalni test 2025-26. Ovakav temporalni split bolje simulira stvarnu upotrebu modela na budućim utakmicama.

Leakage prevencija je ključna: historijski shooter priors računaju se samo iz šuteva prije trenutnog pokušaja. Cold-start slučajevi koriste populacione fallback vrijednosti. Kategorijske varijable obrađuju se kroz pipeline, a numeričke varijable se imputiraju i skaliraju prema modelu.

## 6. Izbor metoda i metrika

Testirane su jednostavne baseline metode, Logistic Regression i HistGradientBoosting. Pošto je cilj vjerovatnoća, primarne metrike su Log Loss, Brier Score, ROC AUC i kalibracija/ECE. Accuracy nije dovoljna jer decision layer koristi numeričku vjerovatnoću u formuli očekivane vrijednosti.

## 7. Treniranje i iteracije

Eksperimenti su išli od globalnog baseline modela, preko shot-value baselinea i generičkog HGB modela, do modela sa sirovim `player_id` i leakage-safe historijskim profilom šutera. Finalna selekcija je HistGradientBoosting sa historical profile + raw ID konfiguracijom i sigmoid kalibracijom.

## 8. Evaluacija finalnog modela

Na 2025-26 regular season test skupu finalni model ima Log Loss 0.63585, Brier 0.22346, ROC AUC 0.66647, stvarni FG% 0.4710 i predviđeni FG% 0.4720. Na 2025-26 clutch subsetu Log Loss je 0.62412, Brier 0.21782, a ROC AUC 0.68176.

Playoff evaluacija pokazuje drift kalibracije, što je važno ograničenje. ROC i confusion matrix su uključeni kao dopunski akademski prikazi, ali model se bira prema vjerovatnosnim metrikama.

## 9. Usporedba modela

| Model | Feature config | Calibration | Log Loss | Brier | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| hist_gradient_boosting | historical_shooter_profile | sigmoid | 0.63579 | 0.22343 | 0.66653 |
| hist_gradient_boosting | historical_profile_plus_raw_id | sigmoid | 0.63585 | 0.22346 | 0.66647 |
| hist_gradient_boosting | generic | sigmoid | 0.63813 | 0.22450 | 0.65964 |
| hist_gradient_boosting | raw_player_id | sigmoid | 0.63880 | 0.22479 | 0.65874 |
| baseline_shot_value | generic | sigmoid | 0.67364 | 0.24036 | 0.59276 |
| baseline_global | generic | sigmoid | 0.69147 | 0.24916 | 0.50000 |

Finalni model je izabran jer daje najbolji balans vjerovatnosne tačnosti, kalibracije i basketball interpretabilnosti.

## 10. Decision Policy — TAKE / REVIEW / DEFER

Decision Policy koristi:

`shot_xpps = p_make * shot_value`

Zatim poredi `shot_xpps` sa replacement vrijednošću. U finalnoj politici korišten je global replacement xPPS 1.080888 i prag ±0.15. Ako je margina iznad praga, preporuka je TAKE; ispod negativnog praga DEFER; unutar opsega REVIEW.

Observed PPS po odluci je deskriptivan rezultat iz historijskih podataka, ne dokaz kauzalnog efekta agenta.

## 11. Dodatni eksperiment: kontekst odbrane

Defanzivni eksperiment koristi 2014/15 shot-level tracking podatke sa closest defender distance, shot clock, dribbles i touch time. V1.1 izabrani model je `V11_A_hgb_regularized_sigmoid` sa Log Loss 0.64436, Brier 0.22696 i ROC AUC 0.64696 na test splitu.

Zbog domain shift problema i razlike između 2014/15 tracking podataka i modernog produkcijskog dataseta, ovaj modul ostaje eksperimentalan. U aplikaciji je support-aware: OOD scenariji ne prikazuju apsolutnu eksperimentalnu vjerovatnoću kao pouzdan rezultat.

## 12. Integracija u aplikaciju

Inference je organizovan kroz Python sklearn servis, tanki .NET Web sloj i Decision Lab UI. Korisnik bira igrača, lokaciju na terenu, kontekst utakmice i dobija produkcijski p_make, xPPS, replacement xPPS i TAKE/REVIEW/DEFER. Advanced Context prikazuje eksperimentalni tracking model bez mijenjanja produkcijske odluke.

## 13. Mogućnosti poboljšanja

Najvažniji nastavci su moderni shot-level tracking podaci, tačniji defender identity u modernim sezonama, ozbiljniji possession/counterfactual model za odbijene šuteve, domain adaptation za playoff i tracking podatke, te produkcijski deployment sa bazom i autentifikacijom.

## 14. Zaključak

NBA Clutch Agent pokazuje puni akademski ML tok: izgradnja dataseta, leakage-safe feature engineering, vjerovatnosno modeliranje, kalibracija, evaluacija i decision policy. Sistem ne tvrdi kauzalnost, nego koristi prediktivnu vjerovatnoću i očekivanu vrijednost kao pomoć u analizi kvaliteta šuta.
