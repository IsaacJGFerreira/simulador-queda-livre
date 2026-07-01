# Simulador de Queda Livre

Simulador educacional de queda livre com e sem resistência do ar.

A proposta é permitir a comparação entre o modelo ideal da Física básica e um modelo mais realista, incluindo:

- resistência do ar;
- densidade atmosférica constante ou variável com a altitude;
- pressão e temperatura;
- vento horizontal e vertical;
- objetos com massa, área frontal e coeficiente de arrasto diferentes;
- gráficos de posição, velocidade, aceleração e trajetória.

## Objetivo didático

Este projeto foi pensado para aulas de Física, principalmente em conteúdos de:

- queda livre;
- segunda lei de Newton;
- força de arrasto;
- velocidade terminal;
- movimento em duas dimensões;
- influência da atmosfera no movimento.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois rode o aplicativo:

```bash
streamlit run app.py
```

## Estrutura inicial

```text
simulador-queda-livre/
├── app.py
├── physics.py
├── atmosphere.py
├── wind.py
├── objects.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Modelos físicos incluídos

### Sem resistência do ar

O movimento é calculado apenas pela aceleração gravitacional.

### Com resistência do ar

A força de arrasto é calculada por:

```text
Fd = 1/2 * rho * Cd * A * v_rel²
```

Onde:

- `rho`: densidade do ar;
- `Cd`: coeficiente de arrasto;
- `A`: área frontal;
- `v_rel`: velocidade relativa entre o objeto e o vento.

## Próximas melhorias planejadas

- Adicionar perfis de planeta: Terra, Lua, Marte e Vênus.
- Adicionar atmosfera em camadas.
- Adicionar animação da queda.
- Exportar tabela da simulação em CSV.
- Adicionar comparação automática entre caso ideal e caso real.
- Criar presets de objetos: bola, pena, paraquedista, gota de chuva e folha de papel.
