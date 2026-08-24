# Two stages, because the toolchain is build-time only. CGO_ENABLED=0 is what
# makes an alpine runtime viable at all: the single direct dependency,
# modernc.org/sqlite, is pure Go, so there is no libsqlite3 to link and no musl
# vs glibc question to answer.
#
# --platform=$BUILDPLATFORM pins the builder to the machine actually running
# the build, and GOOS/GOARCH below cross-compile for the target. Without this,
# a multi-arch build ran the whole Go compile under qemu for the arm64 leg —
# and modernc.org/sqlite (a transpiled C codebase, slow to compile even
# natively) made that the entire publish wall time, paid again on every source
# change. CGO_ENABLED=0 is what makes the cross-compile a pure flag flip; only
# the runtime stage's one-line apk below still runs emulated, and that is
# seconds.
FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS builder
ARG TARGETOS TARGETARCH

WORKDIR /build

# Dependencies in their own layer. go.mod and go.sum churn far less often than
# the source, so most rebuilds reuse this download instead of refetching it.
COPY go.mod go.sum ./
RUN go mod download

# Copy the package directory whole rather than *.go: internal/router/dashboard.html
# is pulled in by go:embed, and a source-only COPY still builds fine on a host
# that has the file while failing inside the image. That asymmetry is the
# failure this layout avoids.
COPY main.go ./
COPY internal/ ./internal/

# -s -w strip the symbol table and DWARF. Nothing reads a Go stack trace off a
# published binary, and the router is deployed as an image far more often than
# it is debugged in one.
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -ldflags="-s -w" -o discrimen .

FROM alpine:3.21

# The router proxies to metered internet providers over TLS. Alpine ships no
# trust store of its own, so without this every https backend fails to verify.
RUN apk add --no-cache ca-certificates

COPY --from=builder /build/discrimen /usr/local/bin/discrimen

# LOG_DB_PATH defaults to /data/llm-router/logs.sqlite, and the tier adapter
# state and the auto-generated persist.key are written beside it. persist.key is
# the one that matters: lose it and every stored endpoint api key becomes
# undecryptable, so the whole of /data has to outlive the container.
VOLUME /data

EXPOSE 8585

# Shell form so ROUTER_PORT is read at container start rather than baked in at
# build time. wget is busybox's, already in the base image; curl would be an
# extra package for a single request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O - "http://127.0.0.1:${ROUTER_PORT:-8585}/health" > /dev/null 2>&1 || exit 1

CMD ["discrimen"]
