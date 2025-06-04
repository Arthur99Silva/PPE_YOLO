# PPEDetectFast 👷🛡️

**Detecção Rápida de Equipamentos de Proteção Individual (EPIs) em Tempo Real usando YOLO.**

Este projeto utiliza o modelo YOLO (You Only Look Once) para detectar Equipamentos de Proteção Individual (EPIs) como capacetes (Hardhat), máscaras (Mask) e coletes de segurança (Safety Vest) em tempo real, a partir de uma webcam ou arquivo de vídeo. O algoritmo também identifica a ausência desses EPIs e outros objetos relevantes em ambientes de construção ou industriais.

## 🌟 Funcionalidades

* **Seleção de Fonte:** Permite ao usuário escolher entre usar a webcam ou carregar um arquivo de vídeo (MP4/AVI) através de uma interface gráfica simples (Tkinter).
* **Aceleração por GPU:** Utiliza GPU (CUDA) automaticamente se disponível, caso contrário, recorre à CPU.
* **Modelo Otimizado:** Carrega um modelo YOLO pré-treinado (`ppe.pt`) e aplica otimizações (`model.fuse()`) para melhor performance.
* **Detecção de Classes Específicas:** Identifica as seguintes classes:
    * `Hardhat` (Capacete)
    * `Mask` (Máscara)
    * `NO-Hardhat` (Sem Capacete)
    * `NO-Mask` (Sem Máscara)
    * `NO-Safety Vest` (Sem Colete de Segurança)
    * `Person` (Pessoa)
    * `Safety Cone` (Cone de Segurança)
    * `Safety Vest` (Colete de Segurança)
    * `machinery` (Maquinário)
    * `vehicle` (Veículo)
* **Visualização Clara:**
    * Desenha caixas delimitadoras coloridas ao redor dos objetos detectados.
        * Verde: EPI correto presente.
        * Vermelho: EPI ausente ou outro objeto.
    * Exibe o rótulo da classe e a confiança da detecção.
* **Baixa Resolução para Performance:** Captura de vídeo em baixa resolução (256x144) para processamento mais rápido, ideal para sistemas com recursos limitados.
* **Janela Redimensionável:** A janela de exibição do vídeo pode ser redimensionada pelo usuário, mantendo a proporção do vídeo.
* **Exibição de FPS:** Mostra a taxa de quadros por segundo (FPS) atual para monitoramento de desempenho.
* **Inferência Otimizada:** Utiliza `half=True` (precisão FP16) durante a inferência para acelerar o processo em GPUs compatíveis.

## ⚙️ Requisitos

* Python 3.x
* OpenCV (`opencv-python`)
* PyTorch (`torch`, `torchvision`, `torchaudio`) - com suporte CUDA se for usar GPU.
* Ultralytics YOLO (`ultralytics`)
* Tkinter (geralmente incluído na instalação padrão do Python)

## 🚀 Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Arthur99Silva/PPE_YOLO.git](https://github.com/Arthur99Silva/PPE_YOLO.git)
    cd PPE_YOLO
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install opencv-python torch torchvision torchaudio ultralytics
    ```
    *Nota: Para suporte CUDA, certifique-se de instalar a versão correta do PyTorch conforme as instruções em [pytorch.org](https://pytorch.org/).*

4.  **Modelo Pré-treinado:**
    * Este script espera um arquivo de modelo chamado `ppe.pt` dentro de um diretório `ConstructionSafety`.
    * Crie o diretório: `mkdir ConstructionSafety`
    * Coloque o seu arquivo `ppe.pt` dentro dele: `ConstructionSafety/ppe.pt`.
        *Se o seu modelo tiver um nome ou caminho diferente, ajuste a linha `model = YOLO("ConstructionSafety/ppe.pt")` no script.*

## 🏃 Como Executar

1.  Navegue até o diretório do projeto.
2.  Execute o script Python:
    ```bash
    python PPEDetectFast.py
    ```
3.  Uma janela aparecerá perguntando se deseja "Usar Webcam" ou "Carregar Vídeo". Selecione a opção desejada.
4.  Se escolher "Carregar Vídeo", uma caixa de diálogo permitirá que você navegue e selecione um arquivo `.mp4` ou `.avi`.
5.  A janela de detecção será aberta, mostrando o vídeo com as detecções.
6.  Pressione a tecla `ESC` para fechar a janela de detecção e encerrar o programa.

## 🛠️ Configuração e Detalhes do Código

* **Dispositivo de Processamento:**
    ```python
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ```
    O script detecta automaticamente a disponibilidade de uma GPU NVIDIA com CUDA.

* **Modelo YOLO:**
    ```python
    model = YOLO("ConstructionSafety/ppe.pt")
    model.fuse()
    model.to(device)
    ```
    Carrega o modelo especificado, otimiza-o fundindo camadas convolucionais e BatchNorm (se aplicável), e o move para o dispositivo selecionado.

* **Classes e Cores:**
    ```python
    classNames = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    ]
    ppe_correto = {'Hardhat', 'Mask', 'Safety Vest'}
    # ... Lógica de coloração baseada nas classes e ppe_correto ...
    ```
    As classes são definidas e um conjunto `ppe_correto` é usado para determinar a cor da caixa delimitadora.

* **Parâmetros de Captura e Inferência:**
    ```python
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
    # ...
    results = model(frame, device=device, half=True, stream=True)
    # ...
    if conf < 0.5:
        continue
    ```
    A captura é configurada para 256x144 pixels. A inferência usa `half=True` (precisão mista FP16) para performance e um limiar de confiança (`conf`) de 0.5 é aplicado para filtrar detecções fracas.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* ou enviar *pull requests*.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes (você pode adicionar um arquivo LICENSE.md com a licença MIT, se desejar).
 
