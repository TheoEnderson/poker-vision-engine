# Poker Analytics Engine (PAE)

Assistente visual para Texas Hold'em: reconhece cartas e valores da mesa, acompanha o ciclo das mãos e apresenta equity, pot odds, EV de referência e recomendações no terminal. Não clica nos botões nem executa ações automaticamente.

Combina OpenCV, Tesseract, Monte Carlo, ranges ponderados, blueprint pré-flop e solver CFR+ pós-flop. As estratégias são aproximações do modelo implementado, não uma garantia de jogo ótimo ou de solução GTO para mesas reais multiway.

## Execução rápida

Com o ambiente instalado e a mesa configurada:

```bash
./venv/bin/python main.py --live
```

Para encaminhar o parâmetro `--loose` ao motor de risco e registrar o desempenho:

```bash
./venv/bin/python main.py --live --loose \
  --profile-jsonl /tmp/pae-latencia.jsonl
```

`--loose` não altera o reconhecimento visual nem configura um novo blueprint pré-flop. Encerre com `Ctrl+C` para imprimir as métricas da sessão.

## Instalação e captura

São necessários Python, Tesseract no `PATH` e um mecanismo de captura compatível com a sessão gráfica. As dependências Python e suas versões mínimas estão em [requirements.txt](requirements.txt).

Exemplo de preparação em Ubuntu/Debian:

```bash
sudo apt install python3 python3-venv tesseract-ocr
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

[src/vision/vision.py](src/vision/vision.py) tenta, nesta ordem:

1. Chrome DevTools Protocol (CDP), em `localhost:9222`.
2. `grim`, quando detecta uma sessão Wayland.
3. Captura do monitor por `mss`, como alternativa.

CDP é opcional e requer uma instância do navegador com depuração remota habilitada. O código procura uma aba ReplayPoker e, na ausência dela, uma aba do tipo `page`. Verifique qual aba está sendo capturada e não exponha a porta de depuração à rede.

A disponibilidade de `grim` depende do compositor Wayland; o fallback por `mss` também depende do ambiente gráfico. Existe caminho de captura por `mss` para Windows, mas as regressões visuais recentes foram reproduzidas no Linux.

Na captura do monitor, janelas sobrepostas podem ocultar cartas, dealer e valores. Mantenha o terminal fora das regiões relevantes. A captura por CDP não inclui essas sobreposições, mas utiliza um ajuste específico de altura para viewports com largura de 1920 px.

Os perfis e as fixtures recentes usam layouts Replay Poker em **1920×1080**. Não há garantia de funcionamento em qualquer resolução, zoom ou disposição de assentos: mudanças podem exigir calibração de ROIs e templates.

## Configuração

[src/config.py](src/config.py) lê variáveis de ambiente e o perfil de [sites.json](sites.json). O perfil padrão é `ReplayPoker`; também existe um perfil `PokerStars`, cuja presença não comprova suporte integral ao cliente.

[.env.example](.env.example) serve como referência. **Não há carregamento automático de `.env`**: exporte as variáveis no shell ou forneça-as ao iniciar o processo.

```bash
HERO_IDENTIFIER=MeuNick POKER_SITE=ReplayPoker \
  ./venv/bin/python main.py --live
```

| Variável | Padrão | Finalidade |
| --- | --- | --- |
| `HERO_IDENTIFIER` | `The_Ment_End` | Nome usado na localização visual do Hero. Ajuste para seu jogador. |
| `POKER_SITE` | `ReplayPoker` | Perfil de ROIs e limiares em `sites.json`. |
| `MONTE_CARLO_ITERATIONS` | `10000` | Número de simulações por recálculo de equity. |
| `POLL_INTERVAL` | `0.5` | Pausa após o processamento de cada captura. |
| `RANK_THRESHOLD` / `SUIT_THRESHOLD` | `0.68` | Limiares de reconhecimento das cartas. |
| `DEALER_THRESHOLD` | `0.52` | Limiar do detector do botão D. |
| `MAX_REALISTIC_POT` | `10000` | Teto configurável de sanidade para o pote. |

As coordenadas podem ser sobrescritas por variáveis como `POT_ROI_TOP`, `POT_ROI_LEFT`, `POT_ROI_WIDTH` e `POT_ROI_HEIGHT`. Para os botões, o prefixo utilizado pelo código é **`ACTION_BUTTONS_ROI_`**.

Alterar apenas ROIs não garante suporte a outro site: símbolos, rótulos, geometria e semântica financeira também precisam ser validados.

## Arquitetura e fluxo

```text
Captura atual
  ├─ Hero e board: segmentação e reconhecimento visual
  └─ Pote, IDs da mão, botões e dealer: tarefas paralelas do mesmo frame
            ↓
Validação do ciclo da mão, cartas e contexto observado
            ↓
Validação financeira e disponibilidade das ações
            ↓
Ranges deduplicados → cache de equity → Monte Carlo quando necessário
            ↓
EV / pot odds → blueprint pré-flop ou solver pós-flop → cache de decisão
            ↓
Painel: recomendação compatível, análise informativa ou suspensão explícita
```

| Componente | Responsabilidade |
| --- | --- |
| [main.py](main.py) | Orquestração live, confirmação de mãos, validação de contexto, caches e painel. |
| [src/vision/](src/vision/) | Captura, localização/classificação das cartas e detecção de adversários ativos. |
| [src/ocr/ocr_reader.py](src/ocr/ocr_reader.py) | Pote, stack, botões, valor de call e dealer/posição. |
| [src/ocr/hand_identity.py](src/ocr/hand_identity.py) | Leitura dos IDs atual/anterior para corroborar a identidade da mão. |
| [src/engine/state_machine.py](src/engine/state_machine.py) | Estado conhecido da mão, transições e validação das cartas. |
| [src/engine/range_model.py](src/engine/range_model.py) | Pesos adversários e aplicação deduplicada das evidências representadas. |
| [src/engine/evaluator.py](src/engine/evaluator.py) | Avaliação das mãos e Monte Carlo com participação real nos empates. |
| [src/engine/risk_engine.py](src/engine/risk_engine.py) | EV de referência, pot odds e integração das estratégias. |
| [src/engine/blueprint.py](src/engine/blueprint.py) | Distribuições de ações pré-flop por tier e contexto implementado. |
| [src/engine/game_tree.py](src/engine/game_tree.py), [src/engine/cfr_solver.py](src/engine/cfr_solver.py) | Árvore financeira e solver pós-flop. |
| [src/diagnostics.py](src/diagnostics.py) | Perfil opcional por etapa e registros JSONL. |
| [src/engine/dataset_logger.py](src/engine/dataset_logger.py) | Registro de análises em `data/dataset_partidas.csv`. |

### Caches e ranges

O cache probabilístico é separado do cache de decisão. Mudanças financeiras podem exigir novo EV e nova estratégia sem repetir Monte Carlo se o contexto probabilístico permanecer igual. O reconhecimento visual e as validações continuam sendo executados: **cache não significa zero CPU**.

Mudanças relevantes, incluindo a quantidade de adversários, invalidam os resultados correspondentes. Uma nova mão confirmada limpa o contexto e os caches anteriores e reinicializa os ranges específicos da mão.

Repetir a mesma evidência de board não reforça cumulativamente os pesos. Uma alteração do pote, por si só, não comprova uma nova ação adversária. O modelo atual não mantém ranges individuais por jogador.

## Segurança e interpretação do painel

- Cartas duplicadas no Hero, no board ou entre ambos impedem a análise matemática daquela captura.
- Leituras temporariamente vazias ou parciais do board não autorizam regressões de street. O último estado válido pode ser preservado, mas não é apresentado como recomendação validada para uma captura incerta.
- Cartas divergentes, ausência do Hero ou mudança financeira isolada não bastam para confirmar uma nova mão. A confirmação combina os sinais disponíveis; capturas ambíguas suspendem a recomendação.
- `bet_to_call == 0` não comprova, sozinho, a disponibilidade de CHECK. A recomendação também é confrontada com o turno e os botões observados.
- Pote ou stack inválidos/não confirmados não são substituídos automaticamente pelo valor anterior, por zero ou por estimativas.
- Botões e IDs possuem reconhecimento de rótulos em português e inglês; isso não implica suporte universal a outros idiomas ou layouts.

### Dealer e posição

O detector verifica o disco do botão D e seu glifo vermelho, reduzindo falsos positivos em símbolos das cartas. Candidatos conflitantes continuam sendo tratados como incertos.

A posição relativa é uma **estimativa geométrica 6-max**, indicada no painel; não reconstrói integralmente a ordem estratégica considerando todos os assentos vazios ou jogadores ausentes.

Uma divergência do dealer não reinicia a mão isoladamente. Com ID consistente, Hero compatível e evolução válida do board, o ciclo pode continuar com posição desconhecida e análise apenas informativa. Duas observações visuais validadas e compatíveis podem recuperar a posição sem limpar ranges ou cache de equity. Sem confirmação suficiente, a recomendação permanece suspensa.

### ALL-IN e limites financeiros

O valor exibido em `ALL-IN` pode representar um total que inclui contribuição anterior, e não o custo incremental. Por exemplo, stack disponível de 1240 e botão `ALL-IN 1250` não autorizam tratar 1250 como call.

Quando custo efetivo, contribuição anterior e pote elegível não estão confirmados, o PAE mantém o call desconhecido e informa **cenário financeiro não suportado**, sem recomendar ALL-IN. Não há modelagem geral de side pots nesta interface.

### Equity, EV e estratégia

A equity é a média da participação do Hero no pote simulado: `1` em vitória exclusiva, `1/k` quando divide a melhor mão com `k` vencedores e `0` em derrota. `P_win`, `P_tie` e `P_lose` representam frequências; em multiway, equity não é necessariamente `P_win + P_tie/2`.

Para pote único elegível, sem rake, o EV líquido de referência do CALL é:

```text
EV_CALL = (equity / 100) × (pote antes do call + custo do call) − custo do call
```

Sem aposta a pagar, o painel mostra a parcela esperada do pote, não o EV estratégico de CHECK com apostas futuras. Esse campo também não é o EV de RAISE ou ALL-IN nem determina sozinho a ação selecionada.

O solver usa uma árvore heads-up simplificada; não representa um jogo multiway completo. A equity multiway correta não elimina essa limitação estratégica.

## Diagnóstico de continuidade e latência

```bash
./venv/bin/python main.py --live --debug-hands \
  --profile-jsonl /tmp/pae-diagnostico.jsonl
```

`--debug-hands` detalha os critérios de continuidade/redistribuição. `--profile-jsonl` registra tempos de captura, Hero, board, dealer, IDs, OCR financeiro, FSM, Monte Carlo, estratégia e saída, além dos motivos das suspensões e do resumo da sessão.

O arquivo JSONL é aberto em modo de acréscimo: pode conter várias sessões. Use caminhos distintos para comparações. Os registros podem conter cartas, IDs e textos reconhecidos; revise-os antes de compartilhar.

O OCR de pote, IDs, botões e dealer roda em quatro tarefas sobre a mesma captura, concluídas antes da atualização da FSM. Os tempos dessas tarefas se sobrepõem e não devem ser somados para estimar a duração do ciclo.

`--poll 0.5` sobrescreve a pausa configurada. O intervalo efetivo entre capturas inclui **processamento + pausa**, portanto não significa duas capturas por segundo. A latência medida até uma recomendação não inclui um instante de início do turno que não tenha sido observado.

Os contadores do terminal e os registros do perfil têm escopos diferentes: capturas fora da vez, por exemplo, podem aparecer no JSONL sem incrementar o total de ciclos processados. Diferencie falhas visuais, suspensões financeiras deliberadas, análises informativas e recomendações validadas.

## Testes

```bash
./venv/bin/python -B -m unittest discover -s tests
git diff --check
```

A suíte em [tests/](tests/) cobre reconhecimento visual/OCR, ciclo de mãos e FSM, contexto financeiro e ações disponíveis, caches, ranges, equity/empates, EV, árvore de jogo, CFR+ e diagnóstico. Fixtures visuais recortadas estão em `tests/fixtures/`, incluindo arquivos PNG codificados em base64.

Referência da última validação de código: **499 testes aprovados no commit `1162603`**. Testes e fixtures não substituem validação ao vivo em outros layouts, resoluções e condições de captura.

## Licença

Distribuído sob a licença [MIT](LICENSE).
