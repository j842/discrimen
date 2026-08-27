package router

import (
	"context"
	"testing"
	"time"
)

// Grader selection consults the matrix per PROMPT, not per worker average.
//
// "Who is strongest overall" and "who is strongest at this kind of question" are
// different questions with different answers, and the judge wants the second:
// the point of grading is to catch a fluent, confident, wrong answer, and the
// worker best placed to see that is the one with evidence on this topic. A
// specialist that is mediocre on average is the right grader for its speciality.
//
// This is also the shape of bug this codebase keeps producing, so the test is
// written to fail if the vector is accepted and then ignored: the two workers are
// arranged so that the per-prompt and the overall answers DISAGREE. Passing nil
// must pick one, passing the vector must pick the other. A graderStrength that
// quietly fell back to summary() would fail the second half.
func TestGraderSelectionUsesThePerPromptPrediction(t *testing.T) {
	topic := []float64{1, 0}
	elsewhere := []float64{0, 1}

	m := newOutcomeMatrix(nil)
	at := time.Now()
	var obs []Observation
	add := func(qid string, vec []float64, backend string, correct bool, n int) {
		m.setVector(qid, vec)
		for i := 0; i < n; i++ {
			obs = append(obs, Observation{
				QID: qid, Backend: backend, Thinking: false, Correct: correct,
				LatencyMS: 1000, Source: obsSourceBench, At: at,
			})
		}
	}
	// "generalist" is strong nearly everywhere and weak on this topic.
	// "specialist" is the reverse. Overall favours the generalist; on THIS prompt
	// the specialist is the one with the evidence.
	for _, qid := range []string{"far-1", "far-2", "far-3", "far-4"} {
		add(qid, elsewhere, "generalist", true, 1)
		add(qid, elsewhere, "specialist", false, 1)
	}
	for _, qid := range []string{"near-1", "near-2", "near-3"} {
		add(qid, topic, "generalist", false, 1)
		add(qid, topic, "specialist", true, 1)
	}
	if err := m.record(context.Background(), obs); err != nil {
		t.Fatalf("record: %v", err)
	}

	reg := newTestRegistry()
	for _, id := range []string{"served", "generalist", "specialist"} {
		reg.upsert(BackendRegistration{ID: id, URL: "http://" + id, Model: "m",
			Features: []string{"chat"}, MaxConcurrency: 1, TTLSeconds: 3600})
		reg.setHealth(id, true, "")
		reg.finishCertification(id, true, map[string]Check{}, 50, 10, "")
	}
	r := &Router{cfg: &Config{}, registry: reg, outcomes: m}
	served := reg.get("served")

	// Sanity: the two questions really do have opposite answers in the matrix.
	if p := m.predict(topic, "specialist", false); !p.known() || p.Correct <= 0.5 {
		t.Fatalf("fixture wrong: specialist on-topic prediction = %+v", p)
	}
	if p := m.predict(topic, "generalist", false); !p.known() || p.Correct >= 0.5 {
		t.Fatalf("fixture wrong: generalist on-topic prediction = %+v", p)
	}

	got, _ := r.judgeGrader(served, nil)
	if got == nil {
		t.Fatal("no grader chosen without a vector")
	}
	if got.ID != "generalist" {
		t.Errorf("no vector: chose %s, want generalist (the best OVERALL rate)", got.ID)
	}

	got, _ = r.judgeGrader(served, topic)
	if got == nil {
		t.Fatal("no grader chosen with a vector")
	}
	if got.ID != "specialist" {
		t.Errorf("with the prompt vector: chose %s, want specialist — the vector is being ignored", got.ID)
	}
}

// classificationVec must not panic on a request the embeddings worker never saw,
// and must keep "no classification" distinguishable from "classified, no vector":
// both fall back to the overall rate, and neither is an error.
func TestClassificationVecHandlesAnUnclassifiedRequest(t *testing.T) {
	if v := classificationVec(nil); v != nil {
		t.Errorf("nil classification gave %v, want nil", v)
	}
	if v := classificationVec(&classification{}); len(v) != 0 {
		t.Errorf("classification with no vector gave %v, want empty", v)
	}
}
