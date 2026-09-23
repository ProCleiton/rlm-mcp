# Modo `exec` opcional — plano de melhoria estrutural

Status: proposal (draft, 2026-09-23). Branch: `feat/exec-mode-planning`
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

## 4. Fase 2 — escopo médio, risco médio (só depois de validar a Fase 1 em uso real)

4. `rlm_exec_async(session_id, code) -> {handle, state}` +
   `rlm_wait(handle, timeout)` / `rlm_status` estendido — desacopla a chamada
   MCP síncrona do tempo real de execução, resolvendo o teto de 120s/step
   sem virar streaming completo. Novo método `SessionManager.exec_async`
   (envia sem aguardar `_drive`, cria `asyncio.Task` guardada em
   `session.jobs`).

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
