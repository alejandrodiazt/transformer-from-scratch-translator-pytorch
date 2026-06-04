# Traductor Sequence-to-Sequence con arquitectura transformer desde cero en PyTorch

Este repositorio contiene la implementación e implementación de un modelo de traducción automática de español a inglés basado en la arquitectura Transformer original de Vaswani et al.. El proyecto fue desarrollado utilizando PyTorch nativo para construir los mecanismos de procesamiento de lenguaje natural, control de máscaras y optimización de tensores sin depender de abstracciones de alto nivel de modelos preentrenados.

## Características del proyecto

- Construcción de arquitectura desde cero, implementación de una clase contenedora `TranslatorTransformer` que integra capas de embedding, codificación posicional y la clase base `nn.Transformer` de PyTorch con configuración orientada a lotes primero (`batch_first=True`).
- Codificación posicional personalizada Desarrollo de un módulo `PositionalEncoding` que utiliza funciones sinusoidales y cosenoidales para inyectar la información de orden secuencial en los embeddings de los tokens.
- Control de máscaras secuenciales implementación de máscaras de relleno (`padding masks`) para ignorar tokens de relleno (`PAD`) y máscaras causales (`causalsubsequent masks`) para prevenir el flujo de información futura en el decodificador durante el entrenamiento autorregresivo.
- Procesamiento de datos eficiente tokenización personalizada con expresiones regulares, construcción de vocabularios indexados con umbrales de frecuencia mínima y carga paralela mediante un objeto `DataLoader` con funciones de colación dinámicas (`collate_fn`).
- Estrategia de optimización avanzada Uso del optimizador Adam combinado con un programador de tasa de aprendizaje de ciclo único (`OneCycleLR`) para ejecutar un calentamiento global de gradientes y prevenir la divergencia del modelo.
- Evaluación métrica monitoreo de la convergencia a través de la pérdida de entropía cruzada con ignorancia del índice de relleno y cálculo automatizado de la métrica BLEU (`Bilingual Evaluation Understudy`) mediante NLTK.

## Arquitectura y parámetros del modelo

El diseño estructural del modelo está definido por los siguientes hiperparámetros de ingeniería de software deep learning[cite 1]

- Dimensión de embedding (`EMBED_DIM`) 256[cite 1]
- Cabezas de atención (`NUM_HEADS`) 8[cite 1]
- Capas de CodificaciónDecodificación (`NUM_LAYERS`) 3[cite 1]
- Dimensión de la red Feed-Forward (`FFN_DIM`) 512[cite 1]
- Longitud máxima de secuencia (`MAX_LEN`) 50 tokens[cite 1]

## Dataset utilizado

El entrenamiento se ejecuta sobre un subconjunto filtrado de OPUS-100 (en-es) extraído de la biblioteca de Hugging Face[cite 1]. El flujo de preparación aplica un filtro estricto para procesar oraciones de longitud intermedia (entre 3 y 25 palabras) distribuidas en[cite 1]
- Conjunto de Entrenamiento 8,000 pares de oraciones[cite 1].
- Conjunto de Validación 1,000 pares de oraciones[cite 1].

## Estructura de archivos generados

Al finalizar la ejecución del script, el pipeline exporta de forma automática los siguientes artefactos[cite 1]
- `logs_entrenamiento.csv` Registro estructurado de la pérdida de entrenamiento y validación por época[cite 1].
- `curva_aprendizaje_traductor.png` Gráfica que visualiza el comportamiento de la entropía cruzada frente a las épocas[cite 1].
- `modelo_traductor_mejorado.pt` Checkpoint del estado del modelo, pesos de la red y mapeo de vocabularios para inferencia futura[cite 1].

## Dependencias requeridas

- Python 3.x
- PyTorch
- Datasets (Hugging Face)
- Matplotlib
- NLTK
- Pandas
- NumPy
