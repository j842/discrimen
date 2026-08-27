package router

// LONG-CONTEXT REASONING — the ability the bank could not report on at all.
//
// Before this file the longest prompt in the whole set was ~1400 estimated
// tokens, so nothing in it exercised a window. That is a hole in the middle of
// what profiling is for: the fleet's workers advertise 32K–256K, routing
// hard-filters on that number, and a strengths-and-weaknesses map that never
// asks a long question cannot report "this worker falls apart at 16K" — it
// reports nothing, which reads as no weakness.
//
// THIS IS NOT contextprobe.go, and the distinction is the whole reason both
// exist. The probe measures the USABLE WINDOW: a needle planted at depth d in a
// haystack of length L, asked back verbatim, laddered until retrieval fails. It
// answers "how much context can this worker retrieve from at all", it is a
// capability probe like the speed probe, and it is deliberately kept out of the
// quality score. What it cannot answer is "can this worker REASON over a long
// input" — aggregate facts scattered through one, order events across one,
// notice that one line contradicts another. A model can pass a needle test at
// 64K and still be unable to add up eleven numbers spread through the same 64K,
// because retrieval is one lookup and this is a hundred. Single-needle lookup is
// therefore the one shape deliberately NOT used below.
//
// THREE TASK FAMILIES, each chosen because ignoring the input cannot produce the
// answer:
//
//	aggregation     add the delta of every fault record. Eleven of them, spread
//	                across the log; the arithmetic is trivial and the search is
//	                the whole difficulty, so this measures attention over the
//	                window rather than mental arithmetic.
//	ordering        nine escalated records carry out-of-order timestamps; name
//	                the keeper of the third-earliest. Requires holding nine
//	                scattered facts at once and sorting them, which no single
//	                lookup can shortcut.
//	contradiction   the roster at the top of the log assigns every unit a keeper;
//	                exactly one record disagrees with it. Requires carrying a
//	                table from the head of the window to its tail.
//
// THREE LENGTHS — ~4K, ~16K and ~48K estimated tokens — because a single length
// measures a threshold and cannot see a slope. A worker that holds at 4K and
// collapses at 16K and one that holds to 48K are different workers to route to,
// and one rung cannot tell them apart. The rungs are ~4x apart so a model has to
// degrade a long way before two rungs read the same.
//
// SYNTHESISED IN CODE, NOT PASTED. A 48K-token prompt is ~192,000 characters and
// has no business being a string literal. More importantly the bank is
// CONTENT-HASHED, in two independent places, and a generator that drifted would
// break both:
//
//	benchQuestionQID (identity.go) hashes the prompt, the expected answer, the
//	match mode and the grader version, and a graded answer is permacached under
//	it. A prompt that differed between two runs of the same binary would MISS on
//	every run — so these nine, the most expensive questions in the bank by two
//	orders of magnitude, would be re-generated on every profile of every worker
//	forever, and no two of those verdicts would be about the same question.
//
//	benchQuestionHash orders questions within a tier by an FNV of Prompt+Expect,
//	and a truncated profile grades the prefix of that order. A prompt that moved
//	would reshuffle the tier and change WHICH questions a budget-stopped run
//	asked, so two profiles of one worker would sit on different exams.
//
// So the generator must be byte-identical on every run, on every machine,
// forever. It therefore uses its own splitmix64 rather than math/rand (whose
// exact stream is a stdlib implementation detail this file must not depend on),
// takes no clock, no map iteration and no environment — the one map it builds is
// a membership set that is never ranged over — and is pinned by golden prompt
// hashes in benchmark_data_longcontext_test.go, so an accidental edit to the
// generator fails CI instead of silently re-asking nine 48K questions.
//
// ─── PROVISIONAL AND UNCALIBRATED TIERS ─────────────────────────────────────
//
// The Tier values below are NOT measurements. The generated half of the bank
// takes its tier from a measured fleet pass-rate band (benchgen_emit.go assigns
// them from calibration data) and the hand-authored half from a charter tuned
// against the live fleet over many revisions (benchmark_data.go); these three are
// an author's ordering of three input lengths, assigned without running a single
// worker, because a calibration is hours of live GPU time. Treat them as
// placeholders that happen to type-check. Before anyone reads a number that
// depends on them:
//
//	go run . bench calibrate -router http://localhost:8585 -token "$ROUTER_ADMIN_KEY"
//
// and re-band from the measured p. Until then the only defensible claim is the
// ordering: 48K is not easier than 16K, which is not easier than 4K.
//
// The tiers are 6, 9 and 10 rather than a new tier 13 for three reasons that are
// constraints rather than preferences:
//
//   - benchWeightedScore buckets tiers 1..10 as "base", 11 as insight and 12+ as
//     coding. A tier 13 would land in the CODING bucket and dilute the twenty
//     points that exist to measure whether a worker can be handed a codebase —
//     the exact fault that moved the headroom band off tier 12 (benchgen_emit.go).
//   - benchTierCategory (benchcategory.go) maps tier -> ability, and a tier
//     missing from it categorises as unknown; 6, 9 and 10 all already map to
//     "reasoning", which is what these measure.
//   - the ladder has to be monotone in input length for the tier to mean
//     anything at all, and 6 < 9 < 10 is.
//
// ─── TWO COSTS, MEASURED, SO NOBODY IS SURPRISED BY THEM ────────────────────
//
// PREFILL. The nine prompts total ~204K estimated tokens, and the no-think pass
// re-asks every question at or above benchHardTier — which is all nine — so a
// full profile prefills ~408K tokens of long context. That is ~90s on
// llm-6000pro's measured 4,665 tok/s and tens of minutes on a CPU worker, where
// the ~48K rung alone will exceed benchAnswerDeadline and score a fail. That
// fail is CORRECT (a worker that cannot prefill 48K inside six minutes cannot
// serve a 48K prompt) but it is a fail earned on throughput, so read it with
// the speed probe beside it.
//
// Two things have since taken the worst case off this bill. A worker whose window
// cannot hold a rung is scored a miss BEFORE dispatch and never prefills it at
// all (see below), and a rung this model has already answered is served from the
// permacache, so the ~408K is a first-sighting cost rather than a per-profile one.
//
// STORAGE. BenchResult carries the full Prompt into the persisted profile, so
// these nine add ~816KB per pass and up to ~1.6MB per worker profile. That is
// the reason there are nine and not ninety.
//
// ─── THE GAP THAT MADE THESE UNMEASURABLE, AND HOW IT CLOSED ────────────────
//
// A worker whose window is smaller than the prompt used not to FAIL these: its
// server rejected the request, benchOne recorded res.errd, and an errored
// question stays out of the denominator by design (runQualityBenchmark), so a
// 32K worker came back silently unmeasured at the 48K rung rather than scored as
// unable to do it — the opposite of what a weakness map should say, and about
// precisely the finding this file exists to produce.
//
// benchOne now judges it before dispatch: a prompt that does not fit the
// worker's usable window is a MISS, counted, with the shortfall in the result
// text. It is judged only when the window is actually KNOWN (usableContextTokens
// returns 0 mid-profile, and guessing a miss from a missing number would fail
// questions a worker can answer), and it is judged BEFORE the permacache is
// consulted, because the window belongs to the deployment while the cache is
// keyed by model — otherwise a 32K deployment would inherit a 128K sibling's
// pass at the 48K rung.

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// longCtxRand is splitmix64, written out here rather than taken from math/rand.
//
// The bank is content-hashed, so "the same question every run" is a correctness
// property, not a convenience. math/rand's exact stream for a given seed is a
// stdlib implementation detail — it has been changed before, and a change would
// silently re-generate every prompt below, giving each one a new qid that no
// permacached verdict answers and a new position in its tier's asking order. Two
// workers profiled a week apart would then be graded on different questions and
// compared on one absolute 0-100 scale. Twelve lines of arithmetic owned by this
// file cannot do that.
type longCtxRand struct{ state uint64 }

func (r *longCtxRand) next() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// intn returns a value in [0,n). Modulo bias is irrelevant here — n is at most a
// few tens of thousands against a 64-bit draw — and rejection sampling would
// make the stream depend on the rejection loop, which is one more thing that
// could drift.
func (r *longCtxRand) intn(n int) int {
	if n <= 0 {
		return 0
	}
	return int(r.next() % uint64(n))
}

// shuffle is Fisher-Yates over an index slice, so every ordering below is drawn
// from the same pinned stream rather than from Go's map iteration order (which
// is deliberately randomised per process and would make the bank differ between
// two runs of the same binary).
func (r *longCtxRand) shuffle(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j := r.intn(i + 1)
		idx[i], idx[j] = idx[j], idx[i]
	}
	return idx
}

// longCtxRoster is the unit -> keeper table printed at the head of every
// document and contradicted by exactly one record in its body. Twelve entries so
// the ordering question can draw nine distinct keepers and still have decoys
// left over.
//
// The names are invented and deliberately unlike any programming language,
// shell or format keyword: benchCategoryOf files a hand-authored question as
// "coding" on a bare language name (benchCodeRe), and a log full of plausible
// systems vocabulary is an easy way to get a long-context reasoning question
// reported as a coding measurement.
var longCtxRoster = []struct{ Unit, Keeper string }{
	{"RX-1042", "HALVARD"},
	{"RX-1043", "OKONKWO"},
	{"RX-1044", "SIGRUN"},
	{"RX-1045", "MBEKI"},
	{"RX-1046", "TAKEDA"},
	{"RX-1047", "LINDQVIST"},
	{"RX-1048", "OYELARAN"},
	{"RX-1049", "VASHCHENKO"},
	{"RX-1050", "ARNASON"},
	{"RX-1051", "DEMBELE"},
	{"RX-1052", "KOVALENKO"},
	{"RX-1053", "NAKASHIMA"},
}

// The routine statuses. "fault" and "escalated" are absent on purpose: they are
// stamped onto chosen records afterwards, so a record can never acquire one by
// accident and give the log two answers to a question that promises one.
var longCtxRoutineStatus = []string{"nominal", "standby", "cleared", "logged"}

// Note prose is assembled from four independent slots. Combinatorially that is
// ~7000 distinct sentences, which matters more than it sounds: a log of a
// thousand near-identical lines is a compression exercise rather than a
// long-context one, and a model that spots the repetition can skim it.
var (
	longCtxSubject = []string{
		"carrier bearing", "standby coupler", "intake damper", "feed regulator",
		"return manifold", "shunt contactor", "drift sensor", "ballast tray",
		"outer seal ring", "trim actuator", "vent louvre", "yoke clamp",
	}
	longCtxVerb = []string{
		"drifted within tolerance", "held steady", "was reseated by hand",
		"read low against the reference", "cycled twice without complaint",
		"was left cold pending the next sweep", "settled after the second pass",
		"required no adjustment",
	}
	longCtxWhen = []string{
		"on the northbound pass", "during the dawn sweep", "at the shift handover",
		"under load", "on the return leg", "while the yard was quiet",
		"after the coupler swap", "ahead of the weekly inspection",
	}
	longCtxTail = []string{
		"duty watch countersigned",
		"no downstream effect observed",
		"entered against the standing order",
		"queue depth unchanged",
		"reference sample retained",
		"noted for the monthly return",
		"cross-checked against the yard board",
		"left open for the incoming shift",
	}
)

// longCtxRecord is one line-pair of the log.
type longCtxRecord struct {
	Index  int
	Unit   string
	Keeper string
	T      int
	Status string
	Delta  int
	Note   string
}

func longCtxRender(rec longCtxRecord) string {
	return fmt.Sprintf("[%04d] unit=%s keeper=%s t=%05d status=%s delta=%+d\n       %s\n",
		rec.Index, rec.Unit, rec.Keeper, rec.T, rec.Status, rec.Delta, rec.Note)
}

// How many records of each kind every document carries, regardless of its
// length. Held CONSTANT across the three rungs on purpose: if the number of
// facts grew with the window, a drop from 4K to 48K would be a harder question
// as well as a longer one, and the rung would stop isolating context length —
// which is the one variable this ladder exists to vary.
const (
	longCtxFaults      = 11
	longCtxEscalations = 9
	longCtxSpecials    = longCtxFaults + longCtxEscalations + 1 // + the roster contradiction
)

// longCtxDoc is one generated log and the ground truth derived from it.
type longCtxDoc struct {
	Body          string
	Records       []longCtxRecord
	FaultSum      int      // answer to the aggregation question
	ThirdKey      string   // keeper of the third-earliest escalated record
	MismatchIndex int      // 1-based index of the record that contradicts the roster
	Options       []string // ordering question's options, in label order
	OptionKey     string   // the option LETTER that is correct
}

// longCtxLabels are the option letters for the ordering question. Ten options
// rather than four, following the tier-9/10 items in benchmark_data.go and
// MMLU-Pro: it drops the guess floor from 25% to 10%, which under pass/fail
// scoring is pure noise removed. "I" is present as an option but is never the
// KEY — benchLetterRe excludes a bare "I" from the last-standalone-letter class
// because it is the pronoun that opens half of all reasoning prose, so a model
// answering a bare "I" would grade as a miss through no fault of its own.
var longCtxLabels = []string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

// longCtxBuild synthesises one document and its answers.
//
// ONE PASS, and the ordering of the two phases is load-bearing. Records are
// generated until the body reaches targetChars, and only THEN are the special
// roles stamped onto chosen slots. Doing it the other way round — deciding the
// roles first — would need the record count up front, which needs the rendered
// length, which needs the roles: a loop that could only be broken by generating
// the stream twice and hoping both passes drew the same numbers. Stamping
// afterwards changes a status token's width by a few characters and nothing
// else, so the finished length lands within a fraction of a percent of target.
func longCtxBuild(seed uint64, targetChars int) longCtxDoc {
	r := &longCtxRand{state: seed}

	var recs []longCtxRecord
	total := 0
	for total < targetChars {
		entry := longCtxRoster[r.intn(len(longCtxRoster))]
		rec := longCtxRecord{
			Index:  len(recs) + 1,
			Unit:   entry.Unit,
			Keeper: entry.Keeper,
			T:      10000 + r.intn(90000),
			Status: longCtxRoutineStatus[r.intn(len(longCtxRoutineStatus))],
			Delta:  r.intn(19) - 9,
			Note: longCtxSubject[r.intn(len(longCtxSubject))] + " " +
				longCtxVerb[r.intn(len(longCtxVerb))] + " " +
				longCtxWhen[r.intn(len(longCtxWhen))] + "; " +
				longCtxTail[r.intn(len(longCtxTail))] + ".",
		}
		total += len(longCtxRender(rec))
		recs = append(recs, rec)
	}
	n := len(recs)

	// One special per bucket over longCtxSpecials equal buckets, so no two land
	// on the same record and every family is spread from the head of the window
	// to its tail. A free random choice would clump, and a clumped aggregation
	// question is answerable from one screenful — which is the context probe's
	// question, not this one. The buckets are computed with the multiply-then-
	// divide form rather than a fixed width so they tile the WHOLE range: at
	// n=96 a width of n/21=4 covers only the first 84 records, leaving the last
	// eighth of the log with nothing in it worth reading.
	slots := make([]int, longCtxSpecials)
	for i := range slots {
		lo, hi := i*n/longCtxSpecials, (i+1)*n/longCtxSpecials
		if hi <= lo {
			hi = lo + 1
		}
		slots[i] = lo + r.intn(hi-lo)
	}
	// The ROLE each slot plays is shuffled, with ONE constraint: the roster
	// contradiction goes in the last third of the buckets.
	//
	// That is deliberate and it is a trade. Left to a free shuffle the mismatch
	// lands 34% and 40% of the way through two of the three shipped rungs, and a
	// model that stopped reading at the halfway mark would still answer both — so
	// the question would not be measuring what this file claims to measure.
	// Constraining it makes "the answer lies past record 2n/3" a property of the
	// construction that longCtxTest can assert, instead of a property of the
	// seeds that happened to be picked. The cost is that a model which somehow
	// knew the construction could narrow its search to the last third of one of
	// nine questions — and it would still have to carry the twelve-line roster
	// from the head of the window to get there, which is the thing being
	// measured.
	mismatchBucket := longCtxSpecials*2/3 + r.intn(longCtxSpecials-longCtxSpecials*2/3)
	roles := make([]int, 0, longCtxSpecials-1)
	for i := 0; i < longCtxFaults; i++ {
		roles = append(roles, 0)
	}
	for i := 0; i < longCtxEscalations; i++ {
		roles = append(roles, 1)
	}
	perm := r.shuffle(len(roles))
	shuffled := make([]int, 0, longCtxSpecials)
	for i, next := 0, 0; i < longCtxSpecials; i++ {
		if i == mismatchBucket {
			shuffled = append(shuffled, 2)
			continue
		}
		shuffled = append(shuffled, roles[perm[next]])
		next++
	}

	// Nine distinct keepers for the nine escalated records: the ordering question
	// answers with a keeper NAME, so two escalated records sharing one would make
	// "the third one" ambiguous in a way no grader could catch.
	rosterPerm := r.shuffle(len(longCtxRoster))
	// Nine distinct timestamps, for the same reason applied to the sort key. Drawn
	// with a seen-set rather than trusting a 90,000-wide range not to collide,
	// because a collision would be rare, silent and would ship.
	times := make([]int, 0, longCtxEscalations)
	seen := map[int]bool{}
	for len(times) < longCtxEscalations {
		t := 10000 + r.intn(90000)
		if seen[t] {
			continue
		}
		seen[t] = true
		times = append(times, t)
	}

	doc := longCtxDoc{}
	escalated := make([]longCtxRecord, 0, longCtxEscalations)
	esc := 0
	for i, slot := range slots {
		switch shuffled[i] {
		case 0:
			// Fault deltas are 1..9, never zero or negative, so the total is a clean
			// two-digit positive number. A negative Expect grades correctly since
			// v35 taught benchNumberRe the sign, but there is no reason to make an
			// answer key exercise that when the question is about the search.
			recs[slot].Status = "fault"
			recs[slot].Delta = 1 + r.intn(9)
			doc.FaultSum += recs[slot].Delta
		case 1:
			entry := longCtxRoster[rosterPerm[esc]]
			recs[slot].Status = "escalated"
			recs[slot].Unit = entry.Unit
			recs[slot].Keeper = entry.Keeper
			recs[slot].T = times[esc]
			escalated = append(escalated, recs[slot])
			esc++
		default:
			// The contradiction: a keeper who is real, and on the roster, but not
			// this unit's. Taking the NEXT roster entry rather than a random one
			// keeps it a single deterministic step from a value already in scope.
			for j, entry := range longCtxRoster {
				if entry.Unit == recs[slot].Unit {
					recs[slot].Keeper = longCtxRoster[(j+1)%len(longCtxRoster)].Keeper
					break
				}
			}
			doc.MismatchIndex = recs[slot].Index
		}
	}
	// The fault records are not counted here on purpose. There is exactly one
	// stamped per bucket that drew role 0, so the count is longCtxFaults by
	// construction — and a counter checked against a constant it was derived from
	// proves nothing. What does prove it is reading the answer back off the page,
	// which is what TestLongContextAnswersAreInTheDocument does.

	// Third-earliest by timestamp. The records were written into the log in
	// arrival order and the timestamps are not, so this cannot be read off the
	// page order — the nine have to be found and then sorted.
	sort.Slice(escalated, func(a, b int) bool { return escalated[a].T < escalated[b].T })
	doc.ThirdKey = escalated[2].Keeper

	// Ten options: the nine escalated keepers plus one decoy from the roster, so
	// "which of these appears at all" is not a shortcut. Shuffled, then the key is
	// moved off label "I" if it landed there — see longCtxLabels.
	opts := make([]string, 0, len(longCtxLabels))
	for _, rec := range escalated {
		opts = append(opts, rec.Keeper)
	}
	for _, e := range longCtxRoster {
		if len(opts) == len(longCtxLabels) {
			break
		}
		used := false
		for _, o := range opts {
			if o == e.Keeper {
				used = true
				break
			}
		}
		if !used {
			opts = append(opts, e.Keeper)
		}
	}
	optPerm := r.shuffle(len(opts))
	ordered := make([]string, len(opts))
	for i, p := range optPerm {
		ordered[i] = opts[p]
	}
	keyAt := 0
	for i, o := range ordered {
		if o == doc.ThirdKey {
			keyAt = i
			break
		}
	}
	if longCtxLabels[keyAt] == "I" {
		other := keyAt - 1
		ordered[keyAt], ordered[other] = ordered[other], ordered[keyAt]
		keyAt = other
	}
	doc.Options = ordered
	doc.OptionKey = longCtxLabels[keyAt]

	var b strings.Builder
	b.WriteString("FIELD LOG — RELAY YARD, SECTOR 7\n\nROSTER (each unit's assigned keeper):\n")
	for _, e := range longCtxRoster {
		fmt.Fprintf(&b, "  %s  keeper=%s\n", e.Unit, e.Keeper)
	}
	b.WriteString("\nRECORDS follow in the order they were RECEIVED, which is not the order " +
		"they occurred in; each record's t field is when it occurred.\n\n")
	for _, rec := range recs {
		b.WriteString(longCtxRender(rec))
	}
	doc.Body = b.String()
	doc.Records = recs
	return doc
}

// longCtxRung is one input length, with the seed that generates it and the tier
// its three questions carry. See the PROVISIONAL banner at the top of the file:
// the tiers are an author's ordering of the lengths, not a measurement.
//
// The seeds are arbitrary and, once shipped, FROZEN. Changing one regenerates
// every prompt and every answer at that rung, and both hashes move with it: the
// three questions get new qids, so every worker's cached verdict for them is
// unreachable and each is re-asked at full price, and benchQuestionHash moves,
// so the order that tier is asked in changes and two truncated profiles sample
// different prefixes. None of that needs a benchmarkVersion bump any more —
// that constant is for a change to the profiling METHOD (see benchmark.go) —
// but it is not free either, and the golden hashes in the test are there to make
// sure it is deliberate.
var longCtxRungs = []struct {
	Seed   uint64
	Chars  int // ~4 chars per token, the same estimate benchPromptTokenEstimate uses
	Tier   int
	Tokens int // nominal rung size, for the comment and the test
}{
	{Seed: 0x5ED1A7F0C0FFEE01, Chars: 16000, Tier: 6, Tokens: 4000},
	{Seed: 0x5ED1A7F0C0FFEE02, Chars: 64000, Tier: 9, Tokens: 16000},
	{Seed: 0x5ED1A7F0C0FFEE03, Chars: 192000, Tier: 10, Tokens: 48000},
}

// longCtxPreamble goes BEFORE the log and the question goes after it. That
// ordering is the measurement: a question stated only up front lets a model
// stream past the log looking for one thing, which is a retrieval test; stated
// afterwards it has to have carried the whole window. The preamble says nothing
// about WHICH question is coming, for the same reason.
const longCtxPreamble = "You will be shown a machine-generated field log, and then asked one question about it. " +
	"The answer depends on records scattered throughout the whole log, so read all of it.\n\n"

// longCtxQuestions builds the nine questions. Called from init() rather than
// written out as literals because a 192,000-character prompt is not a string
// literal, and because a generator is the only form in which "the same bytes
// every run" can be asserted by a test.
func longCtxQuestions() []benchmarkQ {
	out := make([]benchmarkQ, 0, len(longCtxRungs)*3)
	for _, rung := range longCtxRungs {
		doc := longCtxBuild(rung.Seed, rung.Chars)

		// Aggregation. "Give the number only" is in the prompt rather than left to
		// benchRequestFor's append, so the format instruction lands with the
		// question instead of after a 192,000-character document.
		out = append(out, benchmarkQ{
			Tier:  rung.Tier,
			Match: "numeric",
			// The expected sum, not a re-derivation: doc.FaultSum is accumulated from
			// the same records that were rendered, so the key cannot disagree with
			// the page. The test re-derives it by PARSING the rendered body, which is
			// the check that they agree.
			Expect: strconv.Itoa(doc.FaultSum),
			Prompt: longCtxPreamble + doc.Body +
				"\nEvery record above whose status is \"fault\" carries a delta value. " +
				"Add up the delta values of all the fault records. Give the number only.",
		})

		// Ordering.
		var opts strings.Builder
		for i, o := range doc.Options {
			fmt.Fprintf(&opts, "%s) %s\n", longCtxLabels[i], o)
		}
		out = append(out, benchmarkQ{
			Tier:   rung.Tier,
			Match:  "mcq",
			Expect: doc.OptionKey,
			Prompt: longCtxPreamble + doc.Body +
				"\nNine records above have status \"escalated\". Sort those nine by their t value, " +
				"smallest t first. Which keeper filed the THIRD record in that order?\n\n" +
				opts.String() + "\nAnswer with the option letter only.",
		})

		// Contradiction.
		out = append(out, benchmarkQ{
			Tier:   rung.Tier,
			Match:  "numeric",
			Expect: strconv.Itoa(doc.MismatchIndex),
			Prompt: longCtxPreamble + doc.Body +
				"\nThe roster at the top of the log assigns every unit a keeper. Exactly one record " +
				"below the roster names a keeper that disagrees with the roster for that record's unit. " +
				"Give that record's index number (the number in square brackets at the start of the " +
				"record). Give the number only.",
		})
	}
	return out
}

func init() {
	// Package-level vars are fully initialised before any init() runs, so
	// appending here is safe regardless of file order — the same reasoning
	// benchmark_data_live.go's init() records.
	benchmarkQuestions = append(benchmarkQuestions, longCtxQuestions()...)
}
