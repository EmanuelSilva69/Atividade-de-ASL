import cv2
import insightface
from insightface.app import FaceAnalysis

# --- CONFIGURAÇÃO ---
ARQUIVO_MODELO = 'inswapper_128.onnx'
IMG_ORIGEM = 'rosto_origem.jpg'  # Foto da pessoa que você quer COLOCAR
IMG_ALVO = 'foto_alvo.jpg'       # Foto da pessoa que vai SER SUBSTITUÍDA
IMG_SAIDA = 'teste_fake.jpg'     # Nome do arquivo final

# 1. Inicializa o detector de rostos (necessário para achar onde aplicar a troca)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Carrega o modelo de troca (Inswapper)
swapper = insightface.model_zoo.get_model(ARQUIVO_MODELO, download=False, download_zip=False)

# 3. Lê as imagens
img_source = cv2.imread(IMG_ORIGEM) # Quem doa o rosto
img_target = cv2.imread(IMG_ALVO)   # Quem recebe o rosto

# 4. Detecta os rostos nas imagens
faces_source = app.get(img_source)
faces_target = app.get(img_target)

if not faces_source or not faces_target:
    print("ERRO: Não encontrei rosto em uma das imagens.")
    exit()

# Pega o primeiro rosto encontrado em cada imagem (índice 0)
rosto_fonte = faces_source[0]
rosto_alvo = faces_target[0]

# 5. Aplica o Deepfake
# O modelo pega a imagem alvo, substitui o rosto alvo pelas características da fonte
imagem_fake = swapper.get(img_target, rosto_alvo, rosto_fonte, paste_back=True)

# 6. Salva o resultado
cv2.imwrite(IMG_SAIDA, imagem_fake)
print(f"Sucesso! Deepfake salvo em: {IMG_SAIDA}")