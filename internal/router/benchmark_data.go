package router

// benchmarkQuestions grades a worker's quality and — crucially — SPREADS the fleet,
// so difficulty routing has real signal to work with (autoTargetQuality reads the
// score as an absolute 0–100 bar; if every worker scores the same the bar stops
// distinguishing them and routing is pure speed). Quality is a WEIGHTED score (v34/v35):
// tiers 1–10 share 60 points pro-rata, tier 11 shares 20, tier 12+ shares 20 — so a model
// that sweeps 1–10 but can touch neither hard band caps at 60, and the top 40 points are
// earned only where top models actually differ. Each bucket is count-independent, so
// questions can be added freely within a bucket without rescaling (see benchWeightedScore
// in benchmark.go).
//
// Each question is graded in the MODE THE ROUTER SERVES THAT DIFFICULTY IN: the easy
// tiers (below benchHardTier) thinking-off, the hard tiers thinking-on — so a worker is
// measured the way production uses it. This is what keeps a reasoning-first model from
// being under-rated: its hard-tier answers are graded WITH the scratchpad it gets in
// production, not on a thinking-off reflex it never runs at that difficulty.
//
// The catch this creates: with a reasoning budget even a small/quantized model can grind
// out an easy multi-step answer, so a hard-tier question that only needs arithmetic
// SATURATES thinking-on and stops discriminating. The hard tiers therefore lean on what
// a scratchpad doesn't hand you — careful-reading traps, language knowledge, compiler/
// shell semantics — so a weak model still misses them WITH thinking and the fleet stays
// spread. If the top re-bunches at q≈9–10, that's the signal a hard-tier question has
// gone too easy thinking-on and needs replacing.
//
// The twelve tiers are difficulty BANDS, tuned empirically against the live fleet
// (Qwen-27B / Gemma-26B-A4B / Gemma-4-E4B / LFM-1.2B) so each band peels off a quality
// class:
//
//	Tier 1 — controls (thinking-off): every working model passes (sanity floor; a model
//	         that misses these is broken).
//	Tier 2 — floor (thinking-off): the weakest model (≈1B) fails, every competent model
//	         passes.
//	Tier 3 — mid (thinking-on): a careful-reading trap a small model (≈4B) slips on even
//	         with a scratchpad while a strong model reads it correctly.
//	Tier 4 — upper-mid (thinking-on): multi-step arithmetic / code a ≈4B slips on; the
//	         strong models now ace this band, so the real spread lives in tiers 5–8.
//	Tier 5 — hard (thinking-on): number theory and multi-step word problems.
//	Tier 6 — traps (thinking-on): misleading-classic reasoning a model pattern-matches
//	         wrong even thinking-on (the bat-and-ball / kg-of-steel family).
//	Tier 7 — harder (thinking-on): combinatorics, sequences, digit problems.
//	Tier 8 — frontier (thinking-on): modular arithmetic, probability, spatial reasoning —
//	         where even a strong model slips, leaving headroom below a perfect 100.
//	Tier 9 — unrecallable (thinking-on): rules invented in the prompt, priors that are
//	         actively wrong, and other people's reasoning to audit. Nothing here exists in
//	         training data to retrieve.
//	Tier 10 — ceiling (thinking-on): SimpleBench-style world-model traps, where the arithmetic
//	         is a decoy and the answer turns on physical, temporal or social common sense.
//	         (Was the top tier until v33 — a 2026 Qwen3.6-27B now walks through it.)
//	Tier 11 — budget-bounded insight (thinking-on): enumeration problems with a hidden
//	         closed-form shortcut; grinding exceeds the budget, insight answers in ~1k
//	         tokens. The first tier that separates a 27B from a 284B (see the tier's own
//	         comment block for the measured spread and everything that DIDN'T work).
//	         "The budget" was benchThinkMaxTokens at 16384 when this was measured; it is
//	         32768 now, and benchAnswerDeadline's six minutes is what a grinder hits first
//	         on this fleet. The tier still works because the gap it exploits is orders of
//	         magnitude, not a factor of two — but it is the tier most exposed to a raised
//	         ceiling, so re-measure the spread whenever either bound moves.
//	Tier 12 — programming (thinking-on): trace a short program, give its exact output. The
//	         only tier that measures whether a worker can be handed a codebase rather than
//	         a puzzle; every tier above measures reasoning in the abstract. Answers are
//	         facts about the language, so they grade exactly with no execution. Sourced by
//	         abstracting real production bugs (see the tier's own comment block).
//
// THE TIERS ARE SHARED, and the charters above describe only what THIS FILE puts in them.
// Two other files append to benchmarkQuestions at init():
//
//	benchmark_data_live.go   the generated LiveBench half, currently 262 questions across
//	                         tiers 3, 4, 5, 7, 8 and 10 (this file's own 130 and the nine
//	                         below make 401). Their tier is a MEASURED fleet
//	                         pass-rate band with no ability meaning, so an AMC maths item
//	                         and a zebra puzzle sit in the same tier whenever the fleet
//	                         found them equally hard.
//	benchmark_data_longcontext.go  nine synthesised ~4K/16K/48K reasoning-over-a-long-input
//	                         questions, three each in tiers 6, 9 and 10. Their tiers are
//	                         explicitly PROVISIONAL — an author's ordering of the three
//	                         input lengths, never calibrated — and that file says so at
//	                         length.
//
// So a tier's charter tells you what it was BUILT to measure, and benchcategory.go is what
// tells you what a given question actually measures.
//
// WHAT ACTUALLY SPREADS THE TOP, MEASURED. Two whole tiers have been built and thrown away
// getting to the current 9 and 10, and the per-tier numbers off the live fleet
// (Qwen3.6-27B / Granite-8B / Gemma-26B-A4B) are worth keeping because they overturn the
// obvious intuitions:
//
//	tier  6  traps                 100% /  38% / 100%   → 62 points
//	tier  9  unrecallable          100% /  67% /  42%   → 58 points
//	tier  8  frontier maths        100% /  69% /  77%   → 31 points
//	         expert recall         100% /  92% /  83%   → 17 points  (built, measured, cut)
//
// HARDER MATHS DOESN'T WORK. Tiers 5–8 are mostly arithmetic and number theory, and that
// class compresses once thinking is on: reported AIME'25 across the Qwen3 family runs 4B
// 65.6 → 8B 67.3 → 14B 70.4 → 30B-A3B 70.9 → 32B 72.9. Seven points from a 4B to a 32B,
// because a scratchpad is most of what those problems need.
//
// HARDER KNOWLEDGE DOESN'T WORK EITHER. A graduate-science tier in the GPQA-Diamond style
// looked right on paper — knowledge is scratchpad-resistant, and a reasoning budget can't
// supply a fact the model never learned. It scored 17 points of spread and a flat 100% for
// the 27B, and was deleted. Single-fact recall saturates from below far faster than you
// expect; a 2026-era 27B holds the entire undergraduate curriculum. (What makes real
// GPQA-Diamond hard is chaining several specialist facts into multi-step domain reasoning,
// which is a different and much harder thing to author.)
//
// WHAT DOES WORK is anything the model cannot retrieve OR grind out: rules invented in the
// prompt (tier 9) and world-model traps where the obvious computation is a decoy (tiers 6
// and 10). Both top the table, and both are among the cheapest tiers to run.
//
// So the standing rule for anyone adding a tier: don't pick a limit that is the model's
// memory, and don't pick one that is its token budget. Pick one that is its world model.
//
// All questions are ORIGINAL rather than lifted from the benchmarks they're modelled on.
// GPQA in particular is gated behind an agreement not to reproduce its items in plain text,
// precisely to keep them out of training corpora, so copying would both breach that and
// self-contaminate this file. The same reasoning applies to the SimpleBench and
// MisguidedAttention material the trap tiers are built in the spirit of.
//
// The mcq items in tiers 9–10 carry TEN options (A–J), not four, following MMLU-Pro: ten
// choices drop the random-guess floor from 25% to 10% and cut prompt-sensitivity variance
// from 4–5% to ~2%. Under this file's pass/fail percentage scoring, guess-luck is pure noise,
// so the wider option list buys real precision. None of them uses I as the key — see
// benchLetterRe in benchmark.go for why bare "I" can't be read as a pick.
// (Struct benchmarkQ is in benchmark.go.)
var benchmarkQuestions = []benchmarkQ{
	// ---- Tier 1: controls (every working model passes — a miss here means broken) ----
	{Tier: 1, Prompt: "What is 8 + 5?", Expect: "13", Match: "numeric"},
	{Tier: 1, Prompt: "What is the capital of France? Answer with one word.", Expect: "Paris", Match: "contains"},
	{Tier: 1, Prompt: "How many days are in a week?", Expect: "7", Match: "numeric"},
	{Tier: 1, Prompt: "What is the chemical symbol for water? Answer with the formula only.", Expect: "H2O", Match: "contains"},

	// ---- Tier 2: floor (the weakest ~1B model fails; every competent model passes) ----
	// Nothing here is a trap. A trap question graded thinking-off measures one-shot
	// reflex rather than capability — which is why both of the ones that used to sit
	// here now sit in tier 3 (v32).
	{Tier: 2, Prompt: "A man has 7 sons, and each of his sons has exactly one sister. How many children does the man have in total? Give the number only.", Expect: "8", Match: "numeric"},
	{Tier: 2, Prompt: "Who discovered the neutron? Surname only.", Expect: "Chadwick", Match: "contains"},
	{Tier: 2, Prompt: "What is the collective noun for a group of crows? Answer with one word.", Expect: "murder", Match: "contains"},
	{Tier: 2, Prompt: "A hen and a half lays an egg and a half in a day and a half. How many eggs does one hen lay in one day? Give your answer as a fraction.", Expect: "2/3", Match: "contains"},

	// ---- Tier 3: mid, graded THINKING-ON — a careful-reading trap a small model (~4B)
	// pattern-matches wrong even with a scratchpad, while a strong model reads it right.
	// (Plain-arithmetic items here lose their bite thinking-on; see the tier-4 note.) ----
	// "final-contains" (not plain "contains") wherever the expected token appears in the
	// prompt itself: a weak model that merely echoes the subject ("Mary's father's fifth
	// daughter is Lulu") must not pass on the echo — catching that is the whole point of
	// these traps. The token only counts where the answer is asserted (see checkAnswer).
	{Tier: 3, Prompt: "Johnny's mother had four children. The first was named April, the second was named May, and the third was named June. What was the name of the fourth child? Answer with one word.", Expect: "Johnny", Match: "final-contains"},
	// Moved down from tier 2 (v32) to sit beside Johnny, which is the identical trap and
	// has always been tier 3. Unlike the train question below, the gate scores this one
	// BELOW threshold, so production really would serve it thinking-off and a reasoning-
	// first model really would answer "Lulu" — but it is a trap, and tier 2 is the floor
	// band, so the tier label is what's wrong. Expect a new no_think/think disagreement in
	// auditThinkingGate here, exactly like the one Johnny already logs.
	{Tier: 3, Prompt: "Mary's father has five daughters named Lala, Lele, Lili, and Lolo, and one more. What is the name of the fifth daughter? Answer with one word.", Expect: "Mary", Match: "final-contains"},
	{Tier: 3, Prompt: "A bat and a ball cost 1 dollar and 10 cents in total. The bat costs 1 dollar more than the ball. How many cents does the ball cost? Give the number only.", Expect: "5", Match: "numeric"},
	{Tier: 3, Prompt: "A farmer has 15 cows. All but 8 of them die. He then buys 8 more cows. How many cows does he have now? Give the number only.", Expect: "16", Match: "numeric"},
	{Tier: 3, Prompt: "A clock takes 6 seconds to strike 4 o'clock (it chimes 4 times). How many seconds pass between two consecutive chimes? Give the number only.", Expect: "2", Match: "numeric"},
	// Moved down from tier 2 (v32). Two steps — the train clears 200 m, so 2 s — which
	// is a tier-3 shape, not a floor question; the reasoning gate scores it 0.69 and
	// would serve it thinking-ON in production, so grading it thinking-off contradicted
	// this file's own rule and cost a 284B its only non-trap miss (see benchmark.go).
	{Tier: 3, Prompt: "A train 100 metres long travels at 100 metres per second through a tunnel that is 100 metres long. How many seconds pass from when the front of the train enters the tunnel until the rear of the train exits it? Give the number only.", Expect: "2", Match: "numeric"},
	{Tier: 3, Prompt: "Before Mount Everest was discovered, what was the tallest mountain above sea level on Earth? Answer with one word.", Expect: "Everest", Match: "final-contains"},

	// ---- Tier 4: top-end discriminators, graded THINKING-ON. What still splits the fleet
	// WITH a scratchpad: the semantic "a hole holds no soil" trap; and compact C++/bash
	// gotchas that hinge on knowledge a reasoning budget can't supply (unsigned comparison
	// + a newline-stripping subshell; the invalid-octal trap). NOTE: thinking-on, the strong
	// models now ace all of tier 4 (incl. the vec_size question, an all-fail marker only
	// thinking-off), so the real top-end spread has moved to tiers 5–8 below.
	//
	// NOTE: the three plain multi-step arithmetic items below (pages / percentages / wage)
	// were thinking-OFF discriminators; thinking-ON a competent model just solves them, so
	// they no longer spread the fleet on their own — they're kept as easy thinking-on points,
	// while the scratchpad-resistant block at the end of this tier (knowledge gotchas +
	// irrelevant-clause traps, GSM-Symbolic-P2 / GPQA-Diamond class) carries the upper-tier
	// spread. ----
	{Tier: 4, Prompt: "How many cubic metres of soil are inside a hole that is 2 metres deep, 1 metre wide, and 1 metre long? Give the number only.", Expect: "0", Match: "numeric"},
	{Tier: 4, Prompt: "A book has 250 pages. Tom reads 40% of it on Monday and another 30 pages on Tuesday. How many pages are left? Give the number only.", Expect: "120", Match: "numeric"},
	{Tier: 4, Prompt: "What is 15% of 80, plus 20% of 50? Give the number only.", Expect: "22", Match: "numeric"},
	{Tier: 4, Prompt: "A worker earns 12 dollars per hour for the first 8 hours of a shift and 18 dollars per hour for every hour after that. How much does she earn for a 10-hour shift? Give the number only.", Expect: "132", Match: "numeric"},
	{Tier: 4, Prompt: "What does this C++ program print? Give only the output.\n#include <iostream>\nint main(){ std::cout << 5 / 2 * 2.0; }", Expect: "4", Match: "numeric"},
	{Tier: 4, Prompt: "What does this C++ program print? Give only the output.\n#include <iostream>\nint main(){ int a = -1; unsigned b = 1; std::cout << (a < b); }", Expect: "0", Match: "numeric"},
	{Tier: 4, Prompt: "What does this bash script print? Give only the output.\nx=$(printf 'hello\\n\\n\\n')\necho \"${#x}\"", Expect: "5", Match: "numeric"},
	{Tier: 4, Prompt: "What happens when this bash script runs?\nmonth=09\necho $((month + 1))\nA) it prints 10\nB) it prints 9\nC) it fails with an error\nAnswer with just the letter.", Expect: "C", Match: "mcq"},
	{Tier: 4, Prompt: "What does this C++ program print? Give only the output.\n#include <iostream>\n#include <vector>\nint main(){ std::vector<int> v = {1, 2, 3}; std::cout << (v.size() > -1); }", Expect: "0", Match: "numeric"},

	// ---- Tier 4 (added): scratchpad-RESISTANT discriminators — they don't yield to step-by-
	// step working, so they spread the fleet even graded thinking-on. Two kinds: knowledge
	// gotchas (language/compiler/chemistry semantics a reasoning budget can't supply if the
	// model doesn't already know the rule) and GSM-Symbolic-style irrelevant-clause traps (a
	// true-but-useless extra clause that lures a model into over-computing). Answers verified
	// and kept non-negative so the numeric matcher grades them cleanly. ----
	{Tier: 4, Prompt: "What does this C++ program print? Give only the output.\n#include <iostream>\nint main(){ std::cout << (3 == 3 == 3); }", Expect: "0", Match: "numeric"},
	{Tier: 4, Prompt: "What does this C program print? Give only the output.\n#include <stdio.h>\nint main(){ printf(\"%d\", 0.1 + 0.2 == 0.3); }", Expect: "0", Match: "numeric"},
	{Tier: 4, Prompt: "In Python 3, what does the expression 2 ** 3 ** 2 evaluate to? Give the number only.", Expect: "512", Match: "numeric"},
	{Tier: 4, Prompt: "What does this bash script print? Give only the output.\nx=$((2#101 + 1))\necho \"$x\"", Expect: "6", Match: "numeric"},
	{Tier: 4, Prompt: "How many electrons are in a neutral atom of carbon-14?\nA) 14\nB) 8\nC) 6\nD) 12\nAnswer with just the letter.", Expect: "C", Match: "mcq"},
	{Tier: 4, Prompt: "Which chemical element has atomic number 11?\nA) Neon\nB) Sodium\nC) Nitrogen\nD) Lithium\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 4, Prompt: "If today is Wednesday, what day of the week will it be 100 days from now? Answer with the day name.", Expect: "Friday", Match: "contains"},
	{Tier: 4, Prompt: "Liam has 5 boxes, each holding 8 apples. He gives away 3 apples, and notices that 2 of the boxes are painted red. How many apples does Liam have now? Give the number only.", Expect: "37", Match: "numeric"},
	{Tier: 4, Prompt: "A shop sells pens at 3 for 2 dollars. Tom buys 12 pens. The shop floor is tiled in blue. How many dollars does Tom pay? Give the number only.", Expect: "8", Match: "numeric"},

	// ---- Restored hard ladder (tiers 5–8) + the harder tier-4 multi-step items, dropped in
	// the v16 thinking-off rewrite and brought back now that the strong fleet aces tiers 1–4
	// thinking-on. All graded thinking-on (≥ benchHardTier). These are the real spread:
	// number theory, combinatorics, modular arithmetic, probability and misleading-classic
	// reasoning traps a weaker model fails and even a strong one slips on. Answers re-verified;
	// percentage scoring means the larger set needs no rescaling. ----

	// Tier 4 (restored): multi-step arithmetic / code a ~4B slips on but a strong model gets.
	{Tier: 4, Prompt: "A baker makes 3 boxes of 12 cupcakes and 5 boxes of 8 cupcakes. She sells exactly three quarters of all the cupcakes at $2 each. What is her total revenue in dollars?", Expect: "114", Match: "numeric"},
	{Tier: 4, Prompt: "A store buys widgets at $8 each and marks them up by 50%. During a sale it takes 20% off the marked price. If it sells 30 widgets at the sale price, what is the total revenue in dollars?", Expect: "288", Match: "numeric"},
	{Tier: 4, Prompt: "Two trains start 300 km apart on the same track and head toward each other, one at 60 km/h and the other at 90 km/h. How many minutes until they meet?", Expect: "120", Match: "numeric"},
	{Tier: 4, Prompt: "Consider this Python function:\ndef f(n):\n    r = 0\n    for i in range(1, n + 1):\n        if i % 3 == 0 or i % 5 == 0:\n            r += i\n    return r\nWhat does f(20) return?", Expect: "98", Match: "numeric"},
	{Tier: 4, Prompt: "How many distinct arrangements (orderings) are there of the letters in the word BANANA?", Expect: "60", Match: "numeric"},
	{Tier: 4, Prompt: "A car uses 8 litres of fuel per 100 km. On a 350 km trip with fuel at $1.80 per litre, what is the total fuel cost in dollars?", Expect: "50.4", Match: "numeric"},
	{Tier: 4, Prompt: "What is the smaller angle, in degrees, between the hour and minute hands of a clock at exactly 3:15?", Expect: "7.5", Match: "numeric"},

	// Tier 5 (restored): number theory & multi-step word problems.
	{Tier: 5, Prompt: "What is the sum of all three-digit positive integers that are divisible by 7?", Expect: "70336", Match: "numeric"},
	{Tier: 5, Prompt: "A snail is at the bottom of a 30-foot well. Each day it climbs up 3 feet, and each night it slips back 2 feet. On which day does it first reach the top and get out?", Expect: "28", Match: "numeric"},
	{Tier: 5, Prompt: "How many trailing zeros does 100! (100 factorial) have?", Expect: "24", Match: "numeric"},
	{Tier: 5, Prompt: "There is exactly one three-digit number that equals 11 times the sum of its own digits. What is that number?", Expect: "198", Match: "numeric"},
	{Tier: 5, Prompt: "How many integers from 1 to 1000 inclusive are divisible by 3 or by 5 but not by 15?", Expect: "401", Match: "numeric"},
	{Tier: 5, Prompt: "You mix 50 litres of a 30% acid solution with 30 litres of a 70% acid solution. What is the acid concentration of the mixture, as a percentage?", Expect: "45", Match: "numeric"},
	{Tier: 5, Prompt: "What is the remainder when 13 raised to the 99th power is divided by 100? Give the number only.", Expect: "77", Match: "numeric"},

	// Tier 6 (restored + added): misleading-classic reasoning traps a model pattern-matches wrong.
	{Tier: 6, Prompt: "A notebook and a pen cost 220 cents in total. The notebook costs 200 cents more than the pen. How many cents does the pen cost?", Expect: "10", Match: "numeric"},
	{Tier: 6, Prompt: "Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?\nA) 0\nB) 1\nC) 2\nD) 3\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 6, Prompt: "A farmer is at a river with a wolf, a goat, and a cabbage, and a boat that carries the farmer plus one item. He only needs the goat on the far side and does not care what happens to the wolf or cabbage. What is the minimum number of river crossings?\nA) 1\nB) 3\nC) 5\nD) 7\nAnswer with just the letter.", Expect: "A", Match: "mcq"},
	{Tier: 6, Prompt: "Which is heavier?\nA) one kilogram of steel\nB) one feather\nC) they weigh exactly the same\nAnswer with just the letter.", Expect: "A", Match: "mcq"},
	{Tier: 6, Prompt: "A cat that is already dead is sealed in a box with a vial of poison that has a 50% chance of breaking. One hour later, before the box is opened, the cat is:\nA) alive\nB) dead\nC) in a superposition of alive and dead\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 6, Prompt: "You are standing in London facing due west. Is Edinburgh to your left or your right?\nA) left\nB) right\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 6, Prompt: "Start with the number 10. Add 5. Then subtract 3. Then double the result. Then subtract 4. What is the final number?", Expect: "20", Match: "numeric"},
	{Tier: 6, Prompt: "Which is the only U.S. state whose name begins with two vowels? Answer with one word.", Expect: "Iowa", Match: "contains"},

	// Tier 7 (restored + added): combinatorics, sequences, digit problems.
	{Tier: 7, Prompt: "How many integers from 1 to 10000 inclusive are perfect squares but not perfect cubes?", Expect: "96", Match: "numeric"},
	{Tier: 7, Prompt: "How many positive integers less than 1000 have digits that sum to exactly 5?", Expect: "21", Match: "numeric"},
	{Tier: 7, Prompt: "A sequence is defined by a(1) = 3 and a(n+1) = a(n)^2 - 2. What is the value of a(4)? Give the number only.", Expect: "2207", Match: "numeric"},
	{Tier: 7, Prompt: "How many diagonals does a regular dodecagon (12-sided polygon) have? Give the number only.", Expect: "54", Match: "numeric"},
	{Tier: 7, Prompt: "What is the units (last) digit of 3 raised to the power 2026? Give the number only.", Expect: "9", Match: "numeric"},

	// Tier 8 (restored + added): frontier — modular arithmetic, probability, spatial reasoning.
	{Tier: 8, Prompt: "A marble is placed in an empty glass. The glass is turned upside down and set on a table, then picked up and put in a microwave.\nWhere is the marble now?\nA) in the glass\nB) in the microwave\nC) on the table\nAnswer with just the letter.", Expect: "C", Match: "mcq"},
	{Tier: 8, Prompt: "You drive 60 miles at 30 mph, then 60 miles at 60 mph. What is your average speed for the whole 120-mile trip, in mph? Give the number only.", Expect: "40", Match: "numeric"},
	{Tier: 8, Prompt: "Three fair coins are flipped. Given that at least one comes up heads, what is the probability that all three are heads? Answer as a fraction.", Expect: "1/7", Match: "contains"},
	{Tier: 8, Prompt: "A colony of bacteria triples every hour, and its jar is completely full after 12 hours. After how many hours was the jar exactly one-third full?", Expect: "11", Match: "numeric"},
	{Tier: 8, Prompt: "What are the last two digits of 7 raised to the power 2026? Give the number only.", Expect: "49", Match: "numeric"},
	{Tier: 8, Prompt: "What are the last two digits of the sum 1! + 2! + 3! + ... + 100! ? Give the number only.", Expect: "13", Match: "numeric"},
	{Tier: 8, Prompt: "What is the sum of the first 10 prime numbers?", Expect: "129", Match: "numeric"},
	{Tier: 8, Prompt: "How many positive divisors does 2025 have?", Expect: "15", Match: "numeric"},
	{Tier: 8, Prompt: "How many different ways can you make exactly 25 cents using only pennies, nickels, and dimes?", Expect: "12", Match: "numeric"},
	{Tier: 8, Prompt: "A 3x3x3 cube is painted on all six outer faces, then cut into 27 unit cubes. How many unit cubes have exactly two painted faces?", Expect: "12", Match: "numeric"},
	{Tier: 8, Prompt: "What is the remainder when 2 raised to the 100th power is divided by 1000? Give the number only.", Expect: "376", Match: "numeric"},
	{Tier: 8, Prompt: "A man looking at a portrait says: \"Brothers and sisters I have none, but this man's father is my father's son.\" Whose portrait is it?\nA) his own\nB) his son\nC) his father\nD) his nephew\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 8, Prompt: "What is the last digit of 7 raised to the power 7^7 (that is, 7 to the power (7 to the power 7))? Give the number only.", Expect: "3", Match: "numeric"},

	// ---- Tier 9: things that CANNOT be recalled, borrowing the three difficulty levers
	// BBEH (BIG-Bench Extra Hard) isolates that are also cheap in tokens:
	//
	//	learning on the fly     — the rule is DEFINED IN THE PROMPT (novel operators,
	//	                          invented number systems, a fictional grammar), so there is
	//	                          no memorised answer to retrieve and the model must actually
	//	                          execute an unfamiliar procedure without drifting.
	//	going against a strong prior — a modified alphabet, a classic whose famous answer is
	//	                          now wrong. Pattern-matching is not just unhelpful here, it
	//	                          is actively punished, which is what separates a model that
	//	                          reasons from one that retrieves.
	//	finding errors in a reasoning trace — given a worked solution containing one slip,
	//	                          name the first bad step. Auditing a chain is a different
	//	                          skill from producing one, and models are markedly worse at it.
	//
	// What is deliberately NOT copied from BBEH is its LENGTH. Its tasks average roughly
	// seven times the output length of BBH, and its own results section reports models
	// scoring below random because they "could not solve the problem in their effective
	// output token lengths and started degenerating after a point, so no final answer could
	// be extracted." Against benchThinkMaxTokens and benchAnswerDeadline — 16384 tokens and
	// 2 minutes when this tier was written, 32768 and 6 minutes now — that failure mode
	// would truncate every worker alike and measure endurance instead of reasoning: no
	// spread, all cost. The looser bounds have not made it safe, only slower to hit, and the
	// tier-10 lift question (see v36 in benchmark.go) is the worked example of a single
	// under-specified item eating three workers' entire budgets. So every item here is
	// compact: hard to get right, quick to answer, with a short checkable result.
	//
	// Measured spread on this tier: Qwen3.6-27B 12/12, Granite-8B 8/12, Gemma-26B-A4B 5/12 —
	// 58 points, second only to tier 6. It earns its runtime; it just doesn't reach the top. ----

	// Learning on the fly: novel operators. Associativity is stated explicitly because it is
	// the whole difficulty — a model that assumes the usual left-to-right on the second one
	// gets a clean, confident, wrong answer.
	{Tier: 9, Prompt: "Define the operator a # b as follows: if a is even, a # b = a/2 + b; if a is odd, a # b = 3a - b. The operator is LEFT-associative. Compute 8 # 3 # 5 # 2. Give the number only.", Expect: "10", Match: "numeric"},
	{Tier: 9, Prompt: "Define the operator a @ b = 2a - b. Unusually, this operator is RIGHT-associative, so x @ y @ z means x @ (y @ z). Compute 5 @ 3 @ 4 @ 1. Give the number only.", Expect: "11", Match: "numeric"},
	{Tier: 9, Prompt: "Define an operation on the set {0,1,2,3} by a <> b = (a + 2b) mod 4. Compute ((3 <> 1) <> (2 <> 3)) <> 3. Give the number only.", Expect: "3", Match: "numeric"},
	{Tier: 9, Prompt: "In a certain numeral system the digits are written as usual, but the place value of a digit is the FACTORIAL of its position counted from the right, starting at 1! for the rightmost digit. So a string of digits d...d3 d2 d1 has the value (d1 x 1!) + (d2 x 2!) + (d3 x 3!) + ... What is the decimal value of the string 3211 in this system? Give the number only.", Expect: "87", Match: "numeric"},
	// Learning on the fly, linguistic: rules must be checked IN ORDER, and the third rule's
	// condition (a letter count) is the one a skimming model drops.
	{Tier: 9, Prompt: "In a fictional language, the plural of a noun is formed by applying the FIRST of these rules that matches:\n1. If the word ends in a vowel, add -ku.\n2. Otherwise, if the word ends in 'n', replace that final 'n' with -mi.\n3. Otherwise, if the word contains the letter 'a' more than once, double the final consonant and add -a.\n4. Otherwise, add -et.\nWhat is the plural of 'tirek'? Give the word only.", Expect: "tireket", Match: "contains"},

	// Going against a strong prior: alphabetical order is about as strong a prior as a
	// language model has, so re-ranking two letters forces genuine execution over recall.
	// Every option starts with M or T, so the swap cannot be sidestepped.
	{Tier: 9, Prompt: "Consider an alphabet identical to English except that the letters M and T have swapped places in the ordering: T now sorts where M normally does, and M now sorts where T normally does. Every other letter keeps its usual position. Under this ordering, which of these words comes FIRST alphabetically?\nA) match\nB) tiger\nC) mango\nD) table\nE) minor\nF) tulip\nG) medal\nH) thumb\nI) mercy\nJ) trace\nAnswer with just the letter.", Expect: "D", Match: "mcq"},
	// The famous answer (rotations only: (n-1)! = 6) is wrong once reflections are also
	// identified. The trap is the half the model doesn't read.
	{Tier: 9, Prompt: "In how many distinct ways can 4 people be seated around a circular table, if seatings that differ only by a rotation are considered the same AND seatings that differ only by a reflection are also considered the same?\nA) 3\nB) 4\nC) 6\nD) 8\nE) 12\nF) 16\nG) 24\nH) 2\nI) 48\nJ) 1\nAnswer with just the letter.", Expect: "A", Match: "mcq"},
	// A floating ship rises with the tide, so the rung count never changes. The arithmetic
	// the prompt invites (60 cm risen / 30 cm per rung = 2 submerged) is the wrong move.
	{Tier: 9, Prompt: "A rope ladder hangs over the side of a ship that is floating freely at anchor. Its rungs are 30 cm apart. At low tide exactly 10 rungs are above the water. The tide then rises at 15 cm per hour. How many rungs are above the water 4 hours later? Give the number only.", Expect: "10", Match: "numeric"},

	// Finding errors in a reasoning trace: the model must audit someone else's working
	// rather than produce its own. Both traces are correct until one specific step, and in
	// each case the later steps follow consistently from the error — so a model that only
	// checks the conclusion, or only re-solves the problem, names the wrong step.
	{Tier: 9, Prompt: "A student works out the remainder when 3^100 is divided by 7:\nStep 1: 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 (all mod 7).\nStep 2: So the powers of 3 repeat with period 6.\nStep 3: 100 = 6 x 16 + 4.\nStep 4: So 3^100 is congruent to 3^4, which is 5 mod 7.\nStep 5: The remainder is 5.\nWhich is the FIRST step that contains an error? Give the step number only.", Expect: "4", Match: "numeric"},
	{Tier: 9, Prompt: "A student calculates the pH of a 0.10 mol/L solution of a weak acid with Ka = 1.0 x 10^-5:\nStep 1: Set up x^2/(0.10 - x) = 1.0 x 10^-5.\nStep 2: Assume x is much smaller than 0.10, giving x^2/0.10 = 1.0 x 10^-5.\nStep 3: So x^2 = 1.0 x 10^-6.\nStep 4: So x = 1.0 x 10^-3.\nStep 5: pH = -log(1.0 x 10^-3) = 3.\nStep 6: But the acid is weak, so the pH must be above 7; the answer is 7.5.\nWhich is the FIRST step that contains an error? Give the step number only.", Expect: "6", Match: "numeric"},

	// Constraint satisfaction, kept to five entities so it resolves in a few thousand tokens
	// rather than degenerating the way a full BBEH zebra puzzle would.
	{Tier: 9, Prompt: "Five runners - Ana, Ben, Cara, Dev and Eli - finished a race in some order with no ties.\n- Ben finished ahead of exactly two runners.\n- Cara finished immediately after Dev.\n- Ana finished ahead of Ben, but Ana was not first.\n- Eli did not finish last.\nIn what position did Dev finish? Give the position as a number, where 1 means first.", Expect: "4", Match: "numeric"},
	// Compositional: two independent sub-problems fused into one answer, so an error in
	// either half destroys the result and partial credit is impossible.
	{Tier: 9, Prompt: "Let X be the number of trailing zeros of 25! (25 factorial). Let Y be the remainder when 2^40 is divided by 9. What is X multiplied by Y? Give the number only.", Expect: "42", Match: "numeric"},

	// ---- Tier 10: the ceiling — SimpleBench-style world-model traps, the only family this
	// file has ever found that a Qwen3.6-27B does not walk through.
	//
	// WHY THIS FAMILY. Measured per-tier spread across the live fleet (Qwen3.6-27B /
	// Granite-8B / Gemma-26B-A4B) settles the question:
	//
	//	tier  6 (traps)            100% /  38% / 100%   → 62 points
	//	tier  9 (unrecallable)     100% /  67% /  42%   → 58 points
	//	tier  8 (frontier maths)   100% /  69% /  77%   → 31 points
	//	(a recall/knowledge tier lived here briefly and managed 17 points before being cut)
	//
	// The trap family wins, and it is also the cheapest to run — no derivation, just a short
	// scenario and a letter. Tier 6 is the easy end of it (bat-and-ball, kg-of-steel); this
	// tier is the hard end.
	//
	// WHAT MAKES IT HARD, AND WHY IT ISN'T JUST MORE TIER 6. Tier 6 uses MODIFIED CLASSICS,
	// and a 27B now reliably spots the modification — it has seen every variant. SimpleBench's
	// insight is to abandon famous riddles entirely and build FRESH everyday scenarios in
	// which an arithmetic or pattern-matched answer is a decoy, and the real answer turns on a
	// world model: things melt, dissolve and burn out; thrown objects land; a stated fact is a
	// distraction while an unstated physical constraint decides. Nothing can be retrieved,
	// and unlike tier 9 a reasoning budget doesn't help either — the model either simulates
	// the situation or it doesn't. Frontier models reach ~76–82% on SimpleBench against a
	// human baseline of ~84%, so the family still has headroom at the very top; that headroom
	// is the entire point of this tier.
	//
	// THE GUARD. Q8 is deliberately a NON-trap: a tray of ice cubes in a working freezer is
	// still twelve ice cubes. Without it a model could score well on this tier by learning
	// "answer zero / answer the counterintuitive option", which would measure nothing. Any
	// future addition here should preserve that — keep at least one item whose obvious answer
	// is also the correct one.
	//
	// CAVEAT worth knowing: this style is the most SUBJECTIVE in the file. Every item below is
	// written so the intended answer is forced by an explicit clause ("stirs it thoroughly",
	// "the freezer runs normally", "the bridge comfortably holds all three"). If a strong
	// model misses one, read its reasoning before assuming the model is wrong — the question
	// may be, and that is a real failure mode here in a way it never was for arithmetic.
	//
	// The lift question below is the worked example, and the reason the clause is not
	// optional: it originally lacked one (nothing said the lift beats a man running up 30
	// flights) while offering "it depends on the lift's speed" as an option, so the out was
	// defensible and three of four workers spent their whole budget on that single item
	// instead of answering it. Note what the fix took: pinning ONE side still left the
	// 27B answering "it cannot be determined" after 6907 thinking tokens. An item in this
	// tier is only forced when NO quantity is left to judgement — which is exactly what the
	// clean items do (both beanbags are on the ground; the candles' burn time and lighting
	// times are all given). See v36 in benchmark.go. ----
	{Tier: 10, Prompt: "Every minute for five minutes, Ravi drops two sugar cubes into a mug of freshly boiled tea and stirs it thoroughly each time. At the end of the five minutes, how many whole, undissolved sugar cubes are in the mug?\nA) 10\nB) 8\nC) 5\nD) 2\nE) 0\nF) 4\nG) 6\nH) 12\nI) 1\nJ) 20\nAnswer with just the letter.", Expect: "E", Match: "mcq"},
	{Tier: 10, Prompt: "A street performer standing in the middle of an empty, flat car park throws a red beanbag 2 metres straight up, then immediately throws a blue beanbag 4 metres straight up. He then walks away and does not touch either beanbag again. Fifteen minutes later, which beanbag is higher above the ground?\nA) the red one\nB) the blue one\nC) the red one, by 2 metres\nD) the blue one, by 2 metres\nE) the blue one, by 4 metres\nF) they are at the same height\nG) the red one, by 4 metres\nH) it depends on their masses\nI) it depends on air resistance\nJ) there is not enough information\nAnswer with just the letter.", Expect: "F", Match: "mcq"},
	{Tier: 10, Prompt: "You reach a fork in a corridor where two attendants are standing. Both attendants always tell the truth. A large illuminated sign above the left-hand passage reads \"EXIT - THIS WAY\". What is the smallest number of questions you must ask the attendants in order to leave?\nA) 1\nB) 2\nC) 3\nD) 0\nE) 4\nF) 5\nG) 6\nH) 7\nI) 8\nJ) it is impossible to know\nAnswer with just the letter.", Expect: "D", Match: "mcq"},
	{Tier: 10, Prompt: "Three colleagues leave the ground floor of a 30-storey office tower at the same moment, all heading for the roof terrace. Priya takes the express lift, which travels non-stop and reaches the roof terrace in under a minute. Dan takes the stairs, running two at a time, which takes him a little over five minutes. Marcus rides the same express lift as Priya, carrying a full tray of coffees, and is 94 years old. Who reaches the roof terrace last?\nA) Priya\nB) Marcus\nC) Dan\nD) Priya and Marcus, together\nE) Dan and Marcus, together\nF) all three arrive together\nG) Priya and Dan, together\nH) Marcus, because of his age\nI) it cannot be determined\nJ) nobody reaches the roof terrace\nAnswer with just the letter.", Expect: "C", Match: "mcq"},
	{Tier: 10, Prompt: "At the start of every hour, for four hours, Nadia places three fresh birthday candles on a cake and lights them. Each candle burns for about twenty minutes before going out. At the end of the four hours, how many candles on the cake are still alight?\nA) 12\nB) 9\nC) 6\nD) 3\nE) 2\nF) 1\nG) 0\nH) 4\nI) 15\nJ) 20\nAnswer with just the letter.", Expect: "G", Match: "mcq"},
	{Tier: 10, Prompt: "Rosa must choose someone to carry a tray of full wine glasses across a crowded room without spilling any. Tom is 22, a champion sprinter, and has drunk four pints of beer. Wei is 61, walks slowly and carefully, and has drunk only water. Ben is 30, very strong, and has drunk three glasses of wine. Who is most likely to succeed?\nA) Wei\nB) Tom\nC) Ben\nD) Tom, because he is the fastest\nE) Ben, because he is the strongest\nF) all three are equally likely\nG) either Tom or Ben\nH) it cannot be determined\nI) none of them could manage it\nJ) whichever of them is tallest\nAnswer with just the letter.", Expect: "A", Match: "mcq"},
	{Tier: 10, Prompt: "Three hikers need to cross a bridge at night. The bridge comfortably holds all three at once, and a floodlight is permanently mounted on it, so no torch is needed. Walking at their own paces they would take 1, 2 and 5 minutes respectively, and they can walk side by side. What is the minimum total time for all three to get across?\nA) 1\nB) 2\nC) 3\nD) 8\nE) 5\nF) 6\nG) 7\nH) 9\nI) 10\nJ) 17\nAnswer with just the letter.", Expect: "E", Match: "mcq"},
	{Tier: 10, Prompt: "Kofi puts a tray of twelve ice cubes into his freezer at 9am. The freezer runs normally all day and nobody opens it. At 3pm he opens the freezer. How many ice cubes are in the tray?\nA) 0\nB) 1\nC) 2\nD) 3\nE) 6\nF) 9\nG) 11\nH) 12\nI) 13\nJ) 24\nAnswer with just the letter.", Expect: "H", Match: "mcq"},
	{Tier: 10, Prompt: "A shop assistant counts 40 apples into a crate at 8am. During the day she adds 15 more to the crate and sells 22 from it. Separately, a display basket by the till has held 6 apples all day, untouched. At 6pm a driver takes the entire crate away to another branch. How many apples are in the shop at 7pm?\nA) 33\nB) 39\nC) 0\nD) 6\nE) 22\nF) 40\nG) 15\nH) 27\nI) 55\nJ) 11\nAnswer with just the letter.", Expect: "D", Match: "mcq"},
	{Tier: 10, Prompt: "A child sits in a moving car holding the string of a helium balloon that floats freely inside the cabin. All the windows are shut. The driver brakes sharply. Which way does the balloon move relative to the inside of the car?\nA) forward, the same way the passengers lurch\nB) backward, towards the rear of the car\nC) it does not move relative to the car\nD) straight up\nE) straight down\nF) to the left\nG) to the right\nH) it depends on the car's speed\nI) it depends on the balloon's size\nJ) forward first, then backward\nAnswer with just the letter.", Expect: "B", Match: "mcq"},
	{Tier: 10, Prompt: "A flight leaves Auckland at 11pm on Monday and lands in Sydney four hours later. At that time of year Sydney's local time is two hours behind Auckland's. What is the local day and time in Sydney when the flight lands?\nA) 1am Monday\nB) 3am Tuesday\nC) 1am Tuesday\nD) 5am Tuesday\nE) 9pm Monday\nF) 11pm Monday\nG) 3am Monday\nH) 1pm Tuesday\nI) 5am Monday\nJ) 11am Tuesday\nAnswer with just the letter.", Expect: "C", Match: "mcq"},
	{Tier: 10, Prompt: "Amara's friend cancels their dinner plans by text an hour beforehand, for the third time this month. Amara replies: \"Fantastic. I absolutely love it when people do this to me.\" How does Amara most likely feel?\nA) delighted\nB) grateful\nC) indifferent\nD) relieved\nE) confused\nF) excited\nG) proud\nH) amused\nI) surprised\nJ) annoyed\nAnswer with just the letter.", Expect: "J", Match: "mcq"},

	// ---- Tier 11: budget-bounded insight (v33) — the first tier a Qwen3.6-27B fails.
	//
	// WHY THIS EXISTS. The fleet gained a 284B MoE (DeepSeek-V4-Flash) and the bench had
	// nothing left to separate it with: the 27B swept all 97 questions including tier 10
	// (the tier written because it couldn't pass it — a year of model progress closed it),
	// and the 284B's only losses were 2-minute-deadline speed-fails. The fix needed items
	// the 27B actually gets WRONG.
	//
	// WHAT WAS TRIED AND DISCARDED, MEASURED (2026-08-12, Qwen3.6-27B-int4 vs V4-Flash-284B,
	// both thinking-on, 16k budget, no deadline):
	//   - perturbed classics (ignorant-host Monty Hall, left-coin conditional probability,
	//     January-restricted birthday paradox): the entire MisguidedAttention family — the
	//     27B caught every perturbation. 2026 models have absorbed this playbook. 0/10 spread.
	//   - deeper state tracking (dynamic-reference box ops, stack machines with swaps),
	//     compositional fusion, exactly-one-false constraint puzzles, fresh world-model
	//     items: all passed by both. 0/10 spread.
	//   - fresh epistemic sum/product puzzles: brute-force search over bounds/parities found
	//     NO variant with a unique solution besides the memorized classic (4,13). Unusable.
	//
	// WHAT WORKS: problems with a hidden closed-form shortcut where brute enumeration
	// exceeds the thinking budget. The no-banned-digit family has an order-preserving
	// bijection (counting in base b-1); a model that SEES it answers in ~1k tokens, a model
	// that grinds the list dies at the 16k cap with no answer at all. Measured: the 27B
	// found the bijection once (no9-b10, the most famous variant — kept as this tier's
	// guard) and burned out on the other three; the 284B solved all four, fastest in 47s.
	// The two both-fail items measured 994s/1228s even for the 284B — one is kept as the
	// ceiling so a perfect 100 stays out of reach, the other was cut as pure cost.
	//
	// FAMILY RISK, stated honestly: four of five items are digit/base enumeration — one
	// trick family, chosen because it is the ONLY family (of seven tried) that spread this
	// pair. When a future model learns the bijection reflex, this tier saturates like the
	// ten before it and needs the next lever. All items original; answers brute-force
	// verified by script (scratchpad tier11_gen.py pattern), not by hand.
	{Tier: 11, Prompt: "Consider the positive integers that contain no digit 9 in ordinary base-10 notation, listed in increasing order: 1, 2, ..., 8, 10, 11, ... What is the 1000th number in this list? Give the number only.", Expect: "1331", Match: "numeric"},
	{Tier: 11, Prompt: "Consider positive integers whose base-7 representation contains no digit 3. List them in increasing order: 1, 2, 4, 5, 6, 11(base 7)=8, ... What is the 100th such integer, expressed in ordinary base 10? Give the number only.", Expect: "138", Match: "numeric"},
	{Tier: 11, Prompt: "Consider the positive integers whose base-8 (octal) representation contains no digit 5, listed in increasing order. What is the 300th such integer, expressed in ordinary base 10? Give the number only.", Expect: "455", Match: "numeric"},
	{Tier: 11, Prompt: "Consider the positive integers whose base-7 representation contains no digit 3, listed in increasing order. What is the SUM of the first 40 such integers (the sum expressed in ordinary base 10)? Give the number only.", Expect: "1121", Match: "numeric"},
	{Tier: 11, Prompt: "How many positive integers less than 1,000,000 have digits whose PRODUCT is exactly 96? Give the number only.", Expect: "1462", Match: "numeric"},

	// TIER 12 — PROGRAMMING / CODING-AGENT FITNESS (thinking-on).
	//
	// Every tier above measures reasoning in the abstract. None of them answer the
	// question actually being asked when a worker is handed a codebase: can it read code
	// exactly? benchgen.go deliberately excludes LiveBench's coding and agentic_coding
	// categories because they need execution, so this gap does not close by itself.
	//
	// The shape that works here is a TRACE: a short program with one counter-intuitive
	// interaction, answered by its exact output. The answer is a fact about the language,
	// so checkAnswer can grade it exactly and no execution is needed at profiling time.
	//
	// SOURCED FROM REAL BUGS, then abstracted. Each item shares its failure mode with a
	// commit from two months of production work across a Go router, an agent platform, a
	// deploy tool and its bash templates, a Python portal and a Kotlin app — mined for the
	// shape of the trap rather than the code. Nothing here names a real host, repo or
	// service. The recurring class across all of it, and the one this tier keeps hitting:
	// absent/unknown is a distinct third state from negative.
	//
	// WHAT DIDN'T WORK, recorded so it isn't retried. 47 multiple-choice questions were
	// authored alongside these and ALL were cut. In every one the correct option was the
	// longest (chance ~20%): the answer had been written with its full justification and
	// the distractors as one-liners. Measured against the fleet, an 8B with no thinking
	// mode scored 79% on them versus 45% on the traces, and the q94 worker scored 100% —
	// 21 points of spread against the traces' 51. They were measuring option length. If
	// MCQ is ever revisited here, length-match every option first and re-calibrate.
	//
	// CALIBRATION. 95 candidates were graded against 4 live workers spanning q59-q94 and
	// cut by item analysis (D = top-half pass rate minus bottom-half, the same statistic
	// benchgen_emit.go uses); 28 survived with D > 0. Answer keys were verified by
	// EXECUTING every program, which is how the negative-number grader fault in
	// benchNumberRe was found. Measured spread: 39.6 / 68.8 / 87.5 / 97.9 percent, and
	// Spearman rho against the existing q is +1.00 — this tier ranks the fleet the same
	// way tiers 1-11 do, so it is measuring capability rather than an artefact; its value
	// is holding a programming-specific measurement where the general tiers saturate.
	//
	// NOISE, stated honestly: 6 of 33 re-answered cells (18%) flipped verdict at
	// temperature 0. With two workers per half a single flip moves D by 0.50, so the
	// [11..] items are the trustworthy ones and the rest are one flip from D=0. Items
	// whose verdict was directly observed to flip are marked UNSTABLE below. Anything
	// re-calibrated here should be run several times and kept by majority verdict.
	//
	// The per-item comments are: the trap, then p (pass rate) and D (discrimination) with
	// the observed pass pattern across the four workers, best-quality first.
	// deferred functions run after the return VALUE is copied, so mutating a non-named local changes nothing
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc f() int {\n\tn := 0\n\tfor i := 0; i < 3; i++ {\n\t\tdefer func() { n++ }()\n\t}\n\treturn n\n}\n\nfunc main() { fmt.Println(f()) }\n\nGive the number only.", Expect: "0", Match: "numeric"},
	// declaring and assigning on one line makes the declaration's status the exit status, hiding the command's failure
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "What does this bash script print?\n\nf() { local x=$(false); echo \"$?\"; }\nf\n\nGive the number only.", Expect: "0", Match: "numeric"},
	// a nil *T stored in an error interface is not equal to nil; the happy path reports failure
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\ntype MyErr struct{}\n\nfunc (e *MyErr) Error() string { return \"boom\" }\n\nfunc mayFail(ok bool) error {\n\tvar p *MyErr\n\tif !ok {\n\t\tp = &MyErr{}\n\t}\n\treturn p\n}\n\nfunc main() {\n\tn := 0\n\tif mayFail(true) != nil {\n\t\tn++\n\t}\n\tfmt.Println(n)\n}\n\nGive the number only.", Expect: "1", Match: "numeric"},
	// errexit is disabled inside a function used as a condition; the function runs to completion
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "How many lines does this bash script print in total?\n\nset -e\nf() { false; echo reached; }\nif f; then echo yes; else echo no; fi\necho end\n\nGive the number only.", Expect: "3", Match: "numeric"},
	// a narrowing integer conversion wraps silently; Go will not warn and the value is not clamped
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\tvar x int32 = 300\n\tfmt.Println(int8(x))\n}\n\nGive the number only.", Expect: "44", Match: "numeric"},
	// an early-exiting pipe consumer SIGPIPEs the producer; pipefail surfaces it as 141
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "What does this bash script print?\n\nset -o pipefail\nseq 1 200000 | head -1 > /dev/null\necho $?\n\nGive the number only.", Expect: "141", Match: "numeric"},
	// append into spare capacity writes through the shared backing array, mutating the original slice
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\ta := make([]int, 3, 5)\n\ta[0], a[1], a[2] = 1, 2, 3\n\t_ = append(a[:2], 9)\n\tfmt.Println(a[2])\n}\n\nGive the number only.", Expect: "9", Match: "numeric"},
	// defining __eq__ sets __hash__ to None, so the class silently stops being usable in a set or as a dict key
	// p=0.50 D=+1.00 [11..]  UNSTABLE
	{Tier: 12, Prompt: "In Python 3, what happens when this runs? Answer in one short phrase.\n\nclass A:\n    def __eq__(self, other):\n        return True\n\nprint(len({A(), A()}))\n", Expect: "unhashable", Match: "contains"},
	// grep -c prints 0 AND exits 1 on no match, so the || fallback appends a second line
	// p=0.50 D=+1.00 [11..]
	{Tier: 12, Prompt: "What does this bash script print?\n\nn=$(printf 'x\\ny\\n' | grep -c 'ZZZ' || echo 0)\necho \"${#n}\"\n\nGive the number only.", Expect: "3", Match: "numeric"},
	// a comprehension's first iterable is evaluated in the enclosing scope, but its body and conditions cannot see class scope
	// p=0.25 D=+0.50 [1...]
	{Tier: 12, Prompt: "In Python 3, this class body raises NameError. Which LINE raises it? Count the \"class C:\" line as line 1.\n\nclass C:\n    xs = [1, 2, 3]\n    ys = [x * 2 for x in xs]\n    ws = [y for y in range(3) if y in xs]\n\nGive the line number only.", Expect: "4", Match: "numeric"},
	// the exception name is deleted at the end of the except block; inside a function that is UnboundLocalError, not NameError
	// p=0.25 D=+0.50 [1...]
	{Tier: 12, Prompt: "In Python 3, running this raises an exception. Name the exception type exactly (one word).\n\ndef f():\n    try:\n        1 / 0\n    except Exception as e:\n        pass\n    return e\n\nf()\n", Expect: "UnboundLocalError", Match: "contains"},
	// IFS whitespace collapses consecutive delimiters; a legitimately-empty middle field shifts every later field left
	// p=0.25 D=+0.50 [1...]
	{Tier: 12, Prompt: "What does this bash script print?\n\nprintf 'a\\t\\tc\\n' | while IFS=$'\\t' read -r x y z; do echo \"${#x}${#y}${#z}\"; done\n\nGive only the output.", Expect: "110", Match: "numeric"},
	// the `in` operator tests identity before equality, so a NaN already in the list is found but a fresh one is not
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python, this program prints one number. What is it?\n\nn = float('nan')\nvalues = [n]\ncount = 0\nif n in values: count += 1\nif float('nan') in values: count += 1\nprint(count)\n\nGive the number only.", Expect: "1", Match: "numeric"},
	// len() on a string is bytes; converting to []rune gives characters — mixing them corrupts any non-ASCII slice offset
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\ts := \"\\u65e5\\u672c\"\n\tfmt.Println(len(s) + len([]rune(s)))\n}\n\nGive the number only.", Expect: "8", Match: "numeric"},
	// an unquoted empty variable vanishes by word-splitting, leaving a one-argument test that is always true
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "This bash script prints one number. What is it?\n\nv=\"\"\nn=0\nif [ -n $v ]; then n=$((n+1)); fi\nif [ -n \"$v\" ]; then n=$((n+1)); fi\necho \"$n\"\n\nGive the number only.", Expect: "1", Match: "numeric"},
	// strings.TrimLeft takes a cutset, not a prefix — Go's mirror of Python's str.strip trap
	// p=0.75 D=+0.50 [11.1]  UNSTABLE
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport (\"fmt\"; \"strings\")\n\nfunc main() {\n\tfmt.Println(len(strings.TrimLeft(\"filename.tar\", \"fil\")))\n}\n\nGive the number only.", Expect: "9", Match: "numeric"},
	// yielding from a finally block during close() refuses the GeneratorExit; cleanup that awaits or yields is not cleanup
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python 3, this raises an exception. Quote the exception MESSAGE exactly (not the type).\n\ndef g():\n    try:\n        yield 1\n    finally:\n        yield 2\n\nit = g()\nnext(it)\nit.close()\n", Expect: "ignored GeneratorExit", Match: "contains"},
	// len() counts bytes while range counts runes; any offset arithmetic mixing the two corrupts non-ASCII input
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\ts := \"h\\u00e9llo\"\n\tn := 0\n\tfor range s {\n\t\tn++\n\t}\n\tfmt.Println(len(s) + n)\n}\n\nGive the number only.", Expect: "11", Match: "numeric"},
	// os.path.join discards everything before an absolute component, so a base directory is not a sandbox
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python, what does this print?\n\nimport os\nprint(len(os.path.join(\"/var/data/uploads\", \"/etc/passwd\")))\n\nGive the number only.", Expect: "11", Match: "numeric"},
	// list multiplication copies the reference, not the object, so all rows are the same list
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python, this program prints one number. What is it?\n\ngrid = [[]] * 3\ngrid[0].append(1)\nprint(sum(len(row) for row in grid))\n\nGive the number only.", Expect: "3", Match: "numeric"},
	// True, 1 and 1.0 hash and compare equal, so they are one key
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python 3, what does this print?\n\nprint(len({1: 'a', True: 'b', 1.0: 'c'}))\n\nGive the number only.", Expect: "1", Match: "numeric"},
	// cp -a SRC DST copies INTO DST when DST already exists, producing a nested tree a fallback path then deploys
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In bash, directory src/ contains one file f, and an empty directory dst/ already exists. After running:\n\ncp -a src dst\n\nwhat is the full path of file f's copy? Give only the path.", Expect: "dst/src/f", Match: "contains"},
	// lexicographic sort of numerically-indexed names; only breaks once there are 10 or more, which is why it ships
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python, what does this print?\n\nnames = [\"part2\", \"part10\", \"part1\"]\nprint(sorted(names).index(\"part2\"))\n\nGive the number only.", Expect: "2", Match: "numeric"},
	// the right-hand side of a pipeline runs in a subshell, so accumulated state is lost
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "What does this bash script print?\n\nn=0\nprintf 'a\\nb\\nc\\n' | while read -r l; do n=$((n+1)); done\necho \"$n\"\n\nGive the number only.", Expect: "0", Match: "numeric"},
	// comparison operators chain, so `a in b == c` means `(a in b) and (b == c)` — not what it reads like
	// p=0.75 D=+0.50 [111.]  UNSTABLE
	{Tier: 12, Prompt: "In Python 3, what does this print?\n\nprint(int(1 in [1] == True))\n\nGive the number only.", Expect: "0", Match: "numeric"},
	// a sub-slice shares the backing array; appending to it overwrites the parent's later element
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "This Go program prints one number. What is it?\n\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\ts := []int{1, 2, 3, 4}\n\tt := s[1:3]\n\t_ = append(t, 99)\n\tfmt.Println(s[3])\n}\n\nGive the number only.", Expect: "99", Match: "numeric"},
	// printf %d parses a leading zero as octal, so a zero-padded counter or date field silently changes value
	// p=0.75 D=+0.50 [11.1]  UNSTABLE
	{Tier: 12, Prompt: "What does this bash command print?\n\nprintf '%d\\n' 010\n\nGive the number only.", Expect: "8", Match: "numeric"},
	// Python floors toward negative infinity and modulo takes the divisor's sign, unlike C/Go/Java. (+10 keeps the answer positive. That was once a requirement — the numeric grader dropped the sign before comparing — but v35 taught benchNumberRe to keep it, so the offset is now only here to keep the question about floor division rather than about sign handling.)
	// p=0.75 D=+0.50 [111.]
	{Tier: 12, Prompt: "In Python 3, what does this print?\n\nprint((-5 // 2) + (-5 % 2) + 10)\n\nGive the number only.", Expect: "8", Match: "numeric"},
}
