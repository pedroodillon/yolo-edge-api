# Resultados dos Experimentos de Pré-processamento

## Tabela de resultados

| Experimento | Configuração | mAP@0.5 (val) | Δ vs baseline | Observação |
|---|---|---:|---:|---|
| E1-A (baseline) | BGR sem conversão | 0,0095 | — | Referência do experimento de espaço de cor. |
| E1-B | RGB correto | 0,0114 | +0,0019 | A conversão explícita para RGB apresentou resultado superior ao BGR. |
| E2-A | Resize simples (distorção) | 0,0094 | — | Referência do experimento de redimensionamento. |
| E2-B | Letterbox correto | 0,0094 | +0,0000 | Obteve o mesmo mAP do resize simples porque as imagens do dataset são quadradas e não receberam padding. |
| E3-A (baseline) | Sem filtro | 0,0114 | — | Preservou os detalhes da imagem e apresentou o menor custo de pré-processamento. |
| E3-B | GaussianBlur 3 x 3, σ=0,8 | 0,0114 | +0,0000 | Não melhorou o mAP em relação à opção sem filtro e adicionou custo computacional. |
| E3-C | GaussianBlur 5 x 5, σ=1,5 | 0,0098 | -0,0016 | O filtro mais intenso reduziu detalhes e apresentou queda no resultado. |
| E3-D | medianBlur, kernel=3 | 0,0115 | +0,0001 | Apresentou o maior mAP do E3, mas o ganho sobre a opção sem filtro foi muito pequeno. |
| E4-A (baseline) | Sem equalização | 0,0173 | — | Referência do experimento realizado no dataset escurecido com gamma 2,2. |
| E4-B | equalizeHist global | 0,0132 | -0,0041 | A equalização global piorou o mAP e apresentou custo computacional adicional. |
| E4-C | CLAHE, clipLimit=2 e tile=8 | 0,0174 | +0,0001 | Obteve o melhor resultado do E4, justificando seu uso em condições de baixa iluminação. |

## Síntese técnica

Os experimentos avaliaram conversão de espaço de cor, redimensionamento, filtragem e correção de contraste no split de validação do dataset `epi-v1`.

No experimento E1, a conversão explícita de BGR para RGB apresentou mAP@0.5 de 0,0114, contra 0,0095 da manutenção da imagem em BGR. O ganho de 0,0019 sustenta a adoção de `convert_rgb=True` nas três configurações do módulo de pré-processamento.

No experimento E2, o resize simples e o letterbox apresentaram o mesmo mAP@0.5 de 0,0094. Isso ocorreu porque as imagens exportadas pelo Roboflow já possuem resolução quadrada de 640 x 640. Nesse conjunto, o redimensionamento para 416 x 416 manteve a proporção e o letterbox não precisou adicionar padding. Mesmo sem ganho de mAP nesse ensaio, `use_letterbox=True` foi mantido porque os frames reais da câmera podem possuir proporção 4:3 ou 16:9. Nessas entradas, o letterbox preserva a proporção dos objetos e fornece os valores de escala e padding necessários para ajustar as bounding boxes ao espaço da imagem original.

No experimento E3, a configuração sem filtro e o `GaussianBlur` 3 x 3 obtiveram mAP@0.5 de 0,0114. Entretanto, o filtro gaussiano adicionou custo computacional sem oferecer ganho mensurável. O `GaussianBlur` 5 x 5 reduziu o resultado para 0,0098. O `medianBlur` com kernel 3 apresentou o maior resultado do experimento, com mAP@0.5 de 0,0115, mas o ganho de apenas 0,0001 em relação à opção sem filtro não foi suficiente para justificar processamento adicional no caminho padrão. Por isso, o `CONFIG_DEFAULT` permanece sem filtros.

No experimento E4, executado sobre o dataset escurecido artificialmente com gamma 2,2, a condição sem equalização apresentou mAP@0.5 de 0,0173. A equalização global reduziu o resultado para 0,0132 e adicionou custo computacional. O CLAHE aplicado no espaço de cor LAB, com `clipLimit=2` e `tileGridSize=8`, atingiu mAP@0.5 de 0,0174. Embora o ganho tenha sido pequeno, o CLAHE apresentou o melhor resultado do experimento e evita a correção global agressiva. Por isso, foi reservado ao `CONFIG_LOW_LIGHT`, enquanto permanece desativado no `CONFIG_DEFAULT`.

Com base nos resultados, o `CONFIG_DEFAULT` utiliza resolução de inferência 320, conversão para RGB, letterbox e nenhum filtro, priorizando o equilíbrio entre desempenho computacional e robustez. O `CONFIG_LOW_LIGHT` mantém a resolução 320 e ativa o CLAHE em LAB para condições de iluminação adversa. O `CONFIG_HIGH_QUALITY` eleva a resolução para 640 e mantém a conversão RGB e o letterbox para cenários em que a qualidade tem prioridade sobre a velocidade.

Os valores absolutos de mAP devem ser interpretados com cautela porque o modelo disponível nesta etapa é o YOLOv8n original treinado no conjunto COCO, e não um modelo retreinado especificamente para as classes Capacete, Colete e Pessoa. Portanto, as decisões desta atividade foram fundamentadas principalmente nas diferenças relativas entre variantes executadas com o mesmo modelo, dataset e split de validação.
