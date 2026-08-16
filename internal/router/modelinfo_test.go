package router

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// fakeWorker serves the /v1/models and /props bodies a real worker would, and
// records which paths were hit.
func fakeWorker(t *testing.T, models, props string) (*Backend, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body string
		switch req.URL.Path {
		case "/v1/models":
			body = models
		case "/props":
			body = props
		}
		if body == "" {
			http.Error(w, "{}", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	return &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}, srv.Close
}

// The bodies below are trimmed from real fleet responses. The point of the
// metadata is that it survives an --alias: llm-6000pro-deepseek-284B-q8 names
// itself after its dropshell service and claims q8, while the weights it loaded
// are MXFP4.
func TestQueryModelInfoLlamaCppAliased(t *testing.T) {
	b, done := fakeWorker(t,
		`{"data":[{"id":"llm-6000pro-deepseek-284B-q8","object":"model","owned_by":"llamacpp",
		  "meta":{"n_ctx":262144,"n_ctx_train":1048576,"n_params":284334567511,
		          "size":161864270172,"ftype":"MXFP4 MoE"}}]}`,
		`{"model_path":"/models/DeepSeek-V3.2-Exp-UD-Q4_K_XL-00001-of-00005.gguf","total_slots":2}`)
	defer done()

	r := &Router{client: &http.Client{}}
	id, meta := r.queryModelInfo(b)

	if id != "llm-6000pro-deepseek-284B-q8" {
		t.Errorf("id = %q", id)
	}
	if meta.ModelParams != 284334567511 {
		t.Errorf("params = %d", meta.ModelParams)
	}
	if meta.ModelQuant != "MXFP4 MoE" {
		t.Errorf("quant = %q", meta.ModelQuant)
	}
	if meta.ModelSizeBytes != 161864270172 {
		t.Errorf("size = %d", meta.ModelSizeBytes)
	}
	if meta.ModelCtxTrain != 1048576 {
		t.Errorf("ctx_train = %d", meta.ModelCtxTrain)
	}
	if meta.ModelPath != "/models/DeepSeek-V3.2-Exp-UD-Q4_K_XL-00001-of-00005.gguf" {
		t.Errorf("path = %q", meta.ModelPath)
	}
}

// A worker with no /props (vLLM) or an unreadable one must still yield whatever
// /v1/models carried, and must not fabricate a path.
func TestQueryModelInfoNoProps(t *testing.T) {
	b, done := fakeWorker(t, `{"data":[{"id":"Qwen/Qwen3-32B-AWQ","max_model_len":32768}]}`, "")
	defer done()

	r := &Router{client: &http.Client{}}
	id, meta := r.queryModelInfo(b)
	if id != "Qwen/Qwen3-32B-AWQ" {
		t.Errorf("id = %q", id)
	}
	if meta.ModelPath != "" || meta.ModelParams != 0 {
		t.Errorf("expected empty metadata, got %+v", meta)
	}
}

// Older llama.cpp builds put the weights path under default_generation_settings.
func TestQueryModelInfoLegacyPropsPath(t *testing.T) {
	b, done := fakeWorker(t, `{"data":[{"id":"/models/granite-4.1-8b-Q4_K_M.gguf"}]}`,
		`{"default_generation_settings":{"model":"/models/granite-4.1-8b-Q4_K_M.gguf","n_ctx":16384}}`)
	defer done()

	r := &Router{client: &http.Client{}}
	if _, meta := r.queryModelInfo(b); meta.ModelPath != "/models/granite-4.1-8b-Q4_K_M.gguf" {
		t.Errorf("path = %q", meta.ModelPath)
	}
}

// An unreachable worker falls back to the registered model, then the id — the
// behaviour queryModel has always had, since it keys the profile cache.
func TestQueryModelInfoUnreachableFallback(t *testing.T) {
	r := &Router{client: &http.Client{}}

	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: "http://127.0.0.1:1", Model: "declared"}}
	if id, _ := r.queryModelInfo(b); id != "declared" {
		t.Errorf("id = %q, want the declared model", id)
	}

	b.Model = "default" // the placeholder every worker registers with
	if id, _ := r.queryModelInfo(b); id != "w" {
		t.Errorf("id = %q, want the backend id", id)
	}
}

// setModelMeta must never blank a field an earlier probe read: /props 401s and
// vLLM's missing parameter count both arrive as zero values.
func TestSetModelMetaKeepsKnownFields(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "w", 1)

	full := ModelMeta{ModelPath: "/models/x.gguf", ModelParams: 8_000_000_000, ModelQuant: "Q4_K - Medium", ModelSizeBytes: 5_000_000_000, ModelCtxTrain: 131072}
	reg.setModelMeta("w", full)
	reg.setModelMeta("w", ModelMeta{ModelParams: 9_000_000_000})

	got := reg.get("w").ModelMeta
	if got.ModelParams != 9_000_000_000 {
		t.Errorf("params = %d, want the fresh value", got.ModelParams)
	}
	if got.ModelPath != full.ModelPath || got.ModelQuant != full.ModelQuant || got.ModelSizeBytes != full.ModelSizeBytes || got.ModelCtxTrain != full.ModelCtxTrain {
		t.Errorf("unread fields were blanked: %+v", got)
	}
}

func TestDescribeWeightsChange(t *testing.T) {
	base := ModelMeta{ModelPath: "/models/a.gguf", ModelParams: 8e9, ModelQuant: "Q4_K - Medium"}

	if got := describeWeightsChange(base, base); got != "" {
		t.Errorf("identical weights reported a change: %q", got)
	}
	// A probe that read nothing is not evidence of a swap.
	if got := describeWeightsChange(base, ModelMeta{}); got != "" {
		t.Errorf("empty probe reported a change: %q", got)
	}
	swapped := ModelMeta{ModelPath: "/models/b.gguf", ModelParams: 27e9, ModelQuant: "Q8_0"}
	if got := describeWeightsChange(base, swapped); got == "" {
		t.Error("a genuine weights swap went unreported")
	}
}

// niceModelName is what display surfaces (the X-LLM-Backend-Model-Name header)
// show; the fingerprint (Backend.Model) must never be derived from it. The
// real fleet supplies every case here: a vLLM worker with a clean name, a
// llama.cpp worker advertising its file path, and a sharded worker whose
// --alias echoes the worker id and whose real name only exists in /props.
func TestNiceModelName(t *testing.T) {
	cases := []struct {
		name string
		b    Backend
		want string
	}{
		{"clean vLLM name passes through",
			Backend{BackendRegistration: BackendRegistration{ID: "llm-5090", Model: "Qwen3.6-27B-int4-AutoRound"}},
			"Qwen3.6-27B-int4-AutoRound"},
		{"file path reduces to basename without .gguf",
			Backend{BackendRegistration: BackendRegistration{ID: "llm-5090-gemma-26B", Model: "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"}},
			"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"},
		{"alias masking recovered from the weights path, shard counter stripped",
			Backend{BackendRegistration: BackendRegistration{ID: "llm-cpu5060-deepseek-284B-q4", Model: "llm-cpu5060-deepseek-284B-q4"},
				ModelMeta: ModelMeta{ModelPath: "/models/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf"}},
			"DeepSeek-V4-Flash-0731-UD-Q4_K_XL"},
		{"alias masking with no weights path keeps the advertised id",
			Backend{BackendRegistration: BackendRegistration{ID: "w1", Model: "w1"}},
			"w1"},
		{"placeholder model falls back to the path",
			Backend{BackendRegistration: BackendRegistration{ID: "w2", Model: "default"},
				ModelMeta: ModelMeta{ModelPath: "/models/x.gguf"}},
			"x"},
	}
	for _, tc := range cases {
		if got := niceModelName(&tc.b); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}
