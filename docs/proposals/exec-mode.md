# Modo `exec` opcional — plano de melhoria estrutural

Status: Fase 1 implementada; Fase 2 IMPLEMENTADA (branch `feat/exec-async-phase2`, 2026-09-23).
(base: `feat/rlm-over-mcp-core`).
Formato: segue a convenção do repo — `docs/DESIGN.md` com seções numeradas
(citada nas docstrings do código, ex. `server.py` referencia "docs/DESIGN.md,
section 5"). O repo não usa OpenSpec (sem diretório `openspec/`), então este
plano vive em `docs/proposals/` em vez de introduzir uma convenção nova.

## 1. Contexto (achados-base — validados por estudo de código-fonte anterior, sem re-investigação)

`rlm-mcp` é um sandbox Python síncrono (`python3 -I -S agent.py`) para
processamento recursivo de documento longo via fan-out de LLM (paradigma RLM).
Estado atual relevante:

- `SessionManager.exec`/`resume` (`session.py`) bloqueiam em `await _drive(...)`
  até completar/timeout.
- `max_exec_seconds=120` é por CHAMADA; `max_wall_seconds=900` e demais tetos
  são por ÁRVORE (ledger compartilhado, `budget.py`).
- Env do sandbox é scrub por whitelist (`PATH`/`HOME`/`LANG`/`TZ`/`TMPDIR` +
  `RLM_*`, nunca credenciais — DESIGN G3).
- Sem blocklist de imports (subprocess passa, mas contido por
  rlimits + cwd efêmero + killpg no timeout).
- State machine `idle|running|parked|final|dead|closed` não sustenta processo
  persistente (`close` recusa durante `running`, `exec` exige `idle`).

## 2. Proposta: modo `exec` opcional, coexistindo com modo `doc` (default, inalterado)

O modo `doc` (atual) permanece o default com comportamento 100% preservado.
O modo `exec` é opt-in por sessão-canal e destina-se ao caso de uso principal:
execução de comandos/tools com timeout maior e credenciais controladas.
Implementação em 3 fases, em ordem de risco crescente; cada fase só começa
após a anterior estar validada em uso real.

## 3. Fase 1 — escopo pequeno-médio, risco baixo (começar por aqui)

1. `OpenSpec.mode: Literal["doc","exec"] = "doc"` (`types.py`) — novo campo,
   default preserva o comportamento atual 100%.
2. `trusted_env: dict[str,str] | None` (`OpenSpec` em `types.py`) — reinjeção
   controlada de variáveis de ambiente whitelisted por sessão, opt-in, só
   quando `mode="exec"`. `LocalDriver.__init__(..., extra_env)` +
   `_build_sandbox_env(limits, in_fd, out_fd, extra_env)` aplica `scrub_env()`
   e depois `update()` com o `extra_env` validado contra uma allowlist
   explícita de nomes (nunca aceitar wildcard, nunca logar valores).
   `server.py` (`rlm_open`) ganha o parâmetro `trusted_env`, com validação
   de nomes via regex antes de repassar. DESIGN G3 segue intacto: o default
   (`doc`, sem `trusted_env`) nunca vê credenciais.
   Correções pós-simulação real (branch `fix/exec-mode-env-guards`):
   `trusted_env` com chave de `KEEP_ENV` (`PATH`, `HOME`, `LANG`, `TZ`,
   `TMPDIR`) ou prefixo `RLM_` é rejeitado com erro claro ANTES de qualquer
   sessão ser criada — em `server._validate_trusted_env` (fronteira
   `rlm_open`) e replicado em `SessionManager.open` para chamadas diretas
   via `OpenSpec` não passarem pela validação do server; o drop silencioso
   em `_build_sandbox_env` permanece como defesa em profundidade. O env do
   sandbox é scrub por allowlist: `PATH`/`HOME`/etc. visíveis no filho são
   os valores do host herdados via `scrub_env()`, nunca valores arbitrários
   — `trusted_env` não pode alterá-los. `RLIMIT_NPROC`: sessões `exec` usam
   o default mais alto `DEFAULT_RLIMIT_NPROC_EXEC = 2048` (modo `doc`
   continua em 256); a env explícita `RLM_RLIMIT_NPROC` do operador sempre
   vence ambos os defaults. Hosts muito carregados podem ainda precisar
   elevar `RLM_RLIMIT_NPROC` manualmente mesmo com o novo default.
3. Tetos elevados por sessão-canal já são suportados hoje via
   `rlm_open(limits={...})` (`server._merge_limits`) — documentar como usar
   isso no modo `exec` (ex. `max_wall_seconds` alto para builds longos),
   sem mudança de código.

## 4. Fase 2 — IMPLEMENTADA (branch `feat/exec-async-phase2`, 2026-09-23)

`rlm_exec_async(session_id, code) -> {handle, state}` + `rlm_wait(handle,
timeout=30)` — desacopla a chamada MCP síncrona do tempo real de execução,
resolvendo o teto de 120s/step sem virar streaming completo. Design final:

- **Handle = `session_id`.** Só 1 job async por sessão é suportado (a sessão
  fica `running` enquanto o job está pendente, e `exec`/`exec_async` exigem
  `idle` — re-dispatch é recusado in-flow até `wait` coletar o job).
- **Sem estado novo.** A sessão permanece `running` durante o job (decisão
  menos invasiva: nenhum `dispatched` na state machine; todos os guards
  existentes — `exec`/`resume`/`peek`/`close`/`sweep` — valem sem mudanças).
- **Slot único.** `_Session.active_job: asyncio.Task | None` (+ `job_started`
  para o `elapsed` do `pending`) em vez de `dict jobs` — suficiente dado o
  limite de 1 job/sessão; `exec`/`exec_async` compartilham os guards via
  `_prepare_exec` + `_begin_exec_step` (sem drift).
- **`wait(handle, timeout)`:** job pronto → resultado terminal idêntico ao do
  `exec` síncrono (`ok`/`needs_llm`/`final`/`error`/`exhausted`), slot limpo;
  job ainda rodando → `{"status": "pending", "elapsed": X}` SEM tocar no
  estado (ainda `running`) e SEM cancelar (re-`wait` depois); handle
  desconhecido ou já coletado → `SessionError` claro. `needs_llm` via async
  coleta normalmente e o `resume` síncrono existente continua dali (sem
  `rlm_resume_async` — fora de escopo desta fase, como contratado).
- **Orçamento:** a janela ativa do ledger (`resume()`/`pause()`) abre no
  dispatch e fecha quando a Task de fundo completa — nunca no `wait` — de
  modo que o wall clock conta só o tempo real de sandbox ativo, não os gaps
  de poll do harness.
- **Close:** `close()` numa sessão (ou ancestral/descendente) com job
  pendente CANCELA a Task (`task.cancel()`) e fecha o driver (SIGKILL do
  process group + reap — mesmo mecanismo de `_timeout`), sem órfãos; sessão
  `running` em passo *síncrono* continua recusando `close` como antes.
- **Tools:** `rlm_exec_async`/`rlm_wait` delegam a `manager.exec_async`/
  `manager.wait` (sem lógica de budget duplicada no adapter); `rlm_wait`
  valida `timeout` numérico `>= 0` como `invalid_arguments`.
- **NÃO implementado (fora de escopo contratado):** streaming incremental de
  output, `rlm_resume_async`.

## 5. Fase 3 — escopo médio, risco médio (avaliar necessidade real antes de iniciar)

5. Estado `detached-running`/`busy` + tabela de jobs persistentes por sessão,
   para suportar processo tipo watcher/servidor/shell interativo entre
   chamadas — só se houver caso de uso real que justifique (ex. `hub`-like
   dentro do RLM).

## 6. Tabela de escopo/risco

| # | Mudança | Escopo | Risco regressão modo atual |
|---|---|---|---|
| 1 | `OpenSpec.mode="exec"` opt-in | Pequeno-médio | Baixo (default inalterado) |
| 2 | `trusted_env` reinjeção controlada de env whitelisted | Pequeno | Baixo (opt-in) |
| 3 | Uso de `limits` já existente para tetos elevados por sessão-canal | Trivial (já existe) | Baixo |
| 4 | `rlm_exec_async` + `rlm_wait`/poll | Médio (grande com streaming) | Médio |
| 5 | Estado `detached-running` + jobs persistentes | Médio | Médio (mexe na state machine) |
| — | Shell real (`/bin/bash`) no sandbox | Grande | Alto — NÃO recomendado |

## 7. Explicitamente FORA de escopo / não recomendado

Substituir o sandbox Python por shell real (`/bin/bash`) — escopo grande,
risco alto (perde protocolo JSON-lines, rlimits, `_Capture`, trajectory).
Considerado desnecessário: a Fase 1+2 já cobre o caso de uso principal
(execução de comandos/tools com timeout maior e credenciais controladas)
sem reescrever o núcleo do projeto.
