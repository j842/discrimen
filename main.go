// Command discrimen is a self-measuring, OpenAI-compatible LLM router.
//
// It publishes one OpenAI surface over a fleet of downstream endpoints — local
// vLLM and llama.cpp workers, and metered internet providers — and decides for
// itself which one should answer each request. Everything it routes on is
// measured rather than declared: quality, speed, capacity, context and
// capabilities all come from probing the endpoint, not from what the endpoint
// says about itself.
//
// The whole program lives in internal/router; this file exists so that
// `go install github.com/j842/discrimen@latest` produces a binary called
// discrimen.
package main

import "github.com/j842/discrimen/internal/router"

func main() { router.Main() }
