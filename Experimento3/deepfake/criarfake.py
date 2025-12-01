import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# --- CONFIGURAÇÃO ---
# Pega o caminho da pasta onde este script está salvo
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_MODELO = os.path.join(PASTA_ATUAL, 'inswapper_128.onnx')

# Substitua pelos nomes das suas fotos que DEVEM estar na mesma pasta
IMG_ORIGEM = os.path.join(PASTA_ATUAL, 'rosto_origem.jpg') 
IMG_ALVO = os.path.join(PASTA_ATUAL, 'foto_alvo.png')       
IMG_SAIDA = os.path.join(PASTA_ATUAL, 'teste_fake.png')     

print(f"Procurando modelo em: {ARQUIVO_MODELO}")

if not os.path.exists(ARQUIVO_MODELO):
    print("ERRO CRÍTICO: O arquivo 'inswapper_128.onnx' não está na pasta!")
    print("Baixe e coloque ele aqui: ", PASTA_ATUAL)
    exit()

# 1. Inicializa o detector de rostos
# allowed_modules=['detection', 'genderage'] deixa mais leve se der erro de memória
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640)) # ctx_id=-1 força uso de CPU para evitar erro de driver

# 2. Carrega o modelo de troca
swapper = insightface.model_zoo.get_model(ARQUIVO_MODELO, download=False, download_zip=False)

# 3. Lê as imagens
img_source = cv2.imread(IMG_ORIGEM)
img_target = cv2.imread(IMG_ALVO)

if img_source is None or img_target is None:
    print("ERRO: Não encontrei as imagens 'rosto_origem.jpg' ou 'foto_alvo.jpg'.")
    print("Verifique se os arquivos existem e têm exatamente esses nomes.")
    exit()

# 4. Detecta os rostos
faces_source = app.get(img_source)
faces_target = app.get(img_target)

if not faces_source:
    print("ERRO: Não achei rosto na imagem de origem.")
    exit()
if not faces_target:
    print("ERRO: Não achei rosto na imagem alvo.")
    exit()

# Pega o rosto mais proeminente (maior área) para evitar pegar figurantes
rosto_fonte = sorted(faces_source, key=lambda x: x.bbox[2]*x.bbox[3])[-1]
rosto_alvo = sorted(faces_target, key=lambda x: x.bbox[2]*x.bbox[3])[-1]

# 5. Aplica o Deepfake
print("Gerando deepfake... aguarde.")
imagem_fake = swapper.get(img_target, rosto_alvo, rosto_fonte, paste_back=True)

# 6. Salva
cv2.imwrite(IMG_SAIDA, imagem_fake)
print(f"Sucesso! Deepfake salvo em: {IMG_SAIDA}")