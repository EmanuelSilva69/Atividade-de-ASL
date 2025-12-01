import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from PIL import Image
import warnings
import os
import time

# Tenta importar tkinter para janela de seleção de arquivos
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

warnings.filterwarnings('ignore')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
# ==============================================================================
# CONFIGURAÇÕES DE SENSIBILIDADE (CALIBRAGEM V2.1 - WEBCAM)
# ==============================================================================

CONFIG_DETECTOR = {
    'LIMIAR_PICO': 20.0,
    
    # Novos limites baseados na correção do Log-Log
    'LIMIAR_SLOPE_MIN': -1.2, # Se menor que isso: Blur excessivo
    'LIMIAR_SLOPE_MAX': -0.1, # Se maior que isso: Ruído digital puro
    
    'LIMIAR_MID_LOW': 0.05,   # Textura mínima
    'LIMIAR_DESVIO_CRITICO': 0.06 # Kill switch para uniformidade 
}
# ==============================================================================
# --- FUNÇÕES DE CAPTURA LOCAL (SUBSTITUINDO AS DO COLAB) ---

def capturar_multiplos_frames_local(num_frames=3):
    """Captura frames usando a webcam local via OpenCV."""
    frames = []
    cap = cv2.VideoCapture(0) # 0 é geralmente a webcam padrão
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return []

    print(f"\nIniciando captura de {num_frames} frames...")
    print("Olhe para a câmera. Capturando em 3 segundos...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Falha ao capturar frame {i+1}")
            continue
            
        print(f"Frame {i+1}/{num_frames} capturado.")
        
        # OpenCV usa BGR, mas Matplotlib/PIL usam RGB. Vamos converter.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        # Pequena pausa entre frames para variar um pouco
        time.sleep(0.5)

    cap.release()
    #cv2.destroyAllWindows()
    return frames

# --- FUNÇÕES DE PROCESSAMENTO (MANTIDAS IGUAIS) ---

def converter_para_escala_cinza(imagem):
    """
    Converte para escala de cinza de forma robusta, aceitando
    RGB, RGBA, BGR ou imagens que já são cinza.
    """
    if imagem is None:
        raise ValueError("A imagem fornecida é None.")
        
    # Se já for 2D (apenas altura e largura), já é cinza
    if len(imagem.shape) == 2:
        return imagem
        
    # Se tiver canais
    if len(imagem.shape) == 3:
        canais = imagem.shape[2]
        if canais == 3:
            return cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        elif canais == 4:
            return cv2.cvtColor(imagem, cv2.COLOR_RGBA2GRAY)
            
    return imagem

def redimensionar_imagem(imagem, height=512, width=512):
    if imagem is None:
        return np.zeros((height, width), dtype=np.uint8)
    return cv2.resize(imagem, (width, height))

def calcular_espectro_fourier(imagem):
    imagem_float = imagem.astype(np.float32)
    fft_resultado = fft2(imagem_float)
    fft_shifted = fftshift(fft_resultado)
    # Adiciona epsilon pequeno para evitar log(0)
    magnitude = np.log1p(np.abs(fft_shifted) + 1e-10)
    return magnitude, fft_shifted

def extrair_caracteristicas_espectrais(magnitude):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # 1. REMOÇÃO DE DC (Centro)
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    magnitude[R < 3] = 0 # Zera o brilho central
    
    # 2. NORMALIZAÇÃO
    min_val = magnitude.min()
    max_val = magnitude.max()
    div = max_val - min_val
    if div == 0: div = 1
    magnitude_norm = (magnitude - min_val) / div
    
    # 3. PERFIL RADIAL
    r_int = R.astype(int)
    tamanho_max = int(min(h, w) / 2)
    
    soma_radial = np.bincount(r_int.ravel(), weights=magnitude_norm.ravel())
    contagem_radial = np.bincount(r_int.ravel())
    
    limit = min(len(soma_radial), len(contagem_radial), tamanho_max)
    soma_radial = soma_radial[:limit]
    contagem_radial = contagem_radial[:limit]
    
    # Evita divisão por zero
    contagem_radial[contagem_radial == 0] = 1 
    energia_radial = soma_radial / contagem_radial
    
    # 4. SLOPE ESPECTRAL (CORRIGIDO V7)
    # O input já está em escala Log (magnitude). Não aplicamos log no Y de novo.
    # Usamos apenas log no X (Frequência) para o gráfico Log-Linear.
    x_seq = np.arange(1, len(energia_radial))
    y_seq = energia_radial[1:] # Energia já está logarítmica
    
    # Filtra valores muito baixos
    validos = (y_seq > 0) & (x_seq < limit * 0.8) # Ignora as bordas extremas ruidosas
    
    if np.sum(validos) > 10:
        log_freq = np.log(x_seq[validos])
        log_energy = y_seq[validos] # <--- CORREÇÃO: Usamos o valor direto
        
        # Regressão linear
        slope, _ = np.polyfit(log_freq, log_energy, 1)
    else:
        slope = 0

    # 5. BANDAS DE FREQUÊNCIA
    # Definindo faixas baseadas na geometria do rosto
    idx_low_end = int(limit * 0.10)  # Estrutura (0-10%)
    idx_mid_end = int(limit * 0.50)  # Textura (10-50%)
    
    e_low = np.mean(energia_radial[1:idx_low_end]) if idx_low_end > 1 else 0
    e_mid = np.mean(energia_radial[idx_low_end:idx_mid_end]) if idx_mid_end > idx_low_end else 0
    
    # Ratio Textura vs Estrutura
    ratio_mid_low = e_mid / (e_low + 1e-6)

    # 6. MÉTRICAS FINAIS
    pico = np.max(energia_radial) if len(energia_radial) > 0 else 0
    media = np.mean(energia_radial) if len(energia_radial) > 0 else 0
    ratio_pico = pico / (media + 1e-6)

    return {
        'energia_radial': energia_radial,
        'spectral_slope': slope,
        'ratio_pico': ratio_pico,
        'ratio_mid_low': ratio_mid_low,
        'energia_media': media,
        'desvio_padrao_espectral': np.std(magnitude_norm)
    }

def classificar_imagem(imagem):
    imagem_cinza = converter_para_escala_cinza(imagem)
    imagem_redimensionada = redimensionar_imagem(imagem_cinza)
    
    mag, _ = calcular_espectro_fourier(imagem_redimensionada)
    chars = extrair_caracteristicas_espectrais(mag)
    
    score = 0
    detalhes = []
    
    # Configs
    L_SLOPE_MIN = CONFIG_DETECTOR['LIMIAR_SLOPE_MIN']
    L_SLOPE_MAX = CONFIG_DETECTOR['LIMIAR_SLOPE_MAX']
    L_MID = CONFIG_DETECTOR['LIMIAR_MID_LOW']
    L_PICO = CONFIG_DETECTOR['LIMIAR_PICO']
    L_DESVIO = CONFIG_DETECTOR['LIMIAR_DESVIO_CRITICO']
    
    # 1. SLOPE (Agora corrigido)
    slope = chars['spectral_slope']
    if slope < L_SLOPE_MIN:
        score += 3
        detalhes.append(f"[ALERTA] Desfoque Artificial (Slope: {slope:.2f})")
    elif slope > L_SLOPE_MAX:
        score += 4 # Quase fatal, ruído puro
        detalhes.append(f"[SUSPEITO] Ruído Digital (Slope: {slope:.2f})")
    else:
        detalhes.append(f"[OK] Frequência Natural (Slope: {slope:.2f})")
        
    # 2. UNIFORMIDADE (Kill Switch)
    desvio = chars['desvio_padrao_espectral']
    if desvio < L_DESVIO:
        score += 5
        detalhes.append(f"[CRITICO] Assinatura Sintética: {desvio:.4f}")
    elif desvio < 0.08: # Alerta amarelo
        score += 2
        detalhes.append(f"[ALERTA] Espectro muito limpo: {desvio:.4f}")
    else:
        detalhes.append(f"[OK] Variedade: {desvio:.4f}")

    # 3. TEXTURA
    mid = chars['ratio_mid_low']
    if mid < L_MID:
        score += 2
        detalhes.append(f"[ALERTA] Falta de Textura: {mid:.4f}")
    else:
        detalhes.append(f"[OK] Textura: {mid:.4f}")
        
    # 4. PICOS
    if chars['ratio_pico'] > L_PICO:
        score += 2
        detalhes.append(f"[SUSPEITO] Picos Altos: {chars['ratio_pico']:.2f}")
    
    # RESULTADO
    if score >= 4:
        classificacao = "SINTETICA (DEEPFAKE)"
        cor = (255, 0, 0)
        confianca = min(99.9, 60 + (score * 8))
    elif score >= 2:
        classificacao = "SUSPEITA"
        cor = (255, 165, 0)
        confianca = 50.0 + (score * 5)
    else:
        classificacao = "NATURAL"
        cor = (0, 255, 0)
        confianca = max(0, 100 - (score * 20))
        
    return {
        'classificacao': classificacao,
        'confianca': confianca,
        'score': score,
        'cor': cor,
        'detalhes': detalhes,
        'caracteristicas': chars,
        'magnitude': mag,
        'imagem_cinza': imagem_redimensionada
    }
# --- FUNÇÕES DE VISUALIZAÇÃO ---

def visualizar_analise_individual(resultado):
    """Mostra o resultado de uma única imagem (V5.0)."""
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(resultado['imagem_cinza'], cmap='gray')
    ax1.set_facecolor("#c40000")
    
    classificacao = resultado['classificacao']
    confianca = resultado['confianca']
    
    if 'SINTETICA' in classificacao: cor_texto = 'red'
    elif 'SUSPEITA' in classificacao: cor_texto = 'orange'
    else: cor_texto = 'green'
    
    ax1.set_title(f'RESULTADO: {classificacao}\nConfianca: {confianca:.1f}%', 
                  fontsize=12, fontweight='bold', color=cor_texto)
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(resultado['magnitude'], cmap='hot')
    ax2.set_title('ESPECTRO DE FOURIER', fontsize=12, fontweight='bold', color='white')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    char = resultado['caracteristicas']
    
    # ATUALIZADO PARA LER DO CONFIG V5
    metricas_text = f"""
ANALISE V5.0 (SCIENTIFIC)

Classificacao: {resultado['classificacao']}
Confianca: {resultado['confianca']:.1f}%
Score: {resultado['score']}/7

Picos (Grade): {char['ratio_pico']:.2f}
(Limite: {CONFIG_DETECTOR['LIMIAR_PICO']})

Slope (1/f): {char['spectral_slope']:.2f}
(Min: {CONFIG_DETECTOR['LIMIAR_SLOPE_MIN']} / Max: {CONFIG_DETECTOR['LIMIAR_SLOPE_MAX']})

Textura (Mid/Low): {char['ratio_mid_low']:.4f}
(Limite: {CONFIG_DETECTOR['LIMIAR_MID_LOW']})

Desvio (Veto): {char['desvio_padrao_espectral']:.4f}
(Critico: {CONFIG_DETECTOR['LIMIAR_DESVIO_CRITICO']})
"""
    
    for detalhe in resultado['detalhes']:
        metricas_text += f"\n{detalhe}"
    
    ax3.text(0.05, 0.95, metricas_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9), color='#00ff00')
    
    plt.tight_layout()
    print("Feche a janela do gráfico para continuar...")
    plt.show()

def visualizar_multiplos_frames(frames, resultados):
    n_frames = len(frames)
    if n_frames == 0: return

    fig = plt.figure(figsize=(18, 4*n_frames))
    fig.patch.set_facecolor('#0a0a0a')
    
    for idx, (frame, resultado) in enumerate(zip(frames, resultados)):
        ax1 = plt.subplot(n_frames, 3, idx*3 + 1)
        ax1.imshow(resultado['imagem_cinza'], cmap='gray')
        ax1.set_facecolor('#1a1a1a')
        
        classificacao = resultado['classificacao']
        confianca = resultado['confianca']
        cor = 'red' if 'SINTETICA' in classificacao else ('orange' if 'INDETERMINADO' in classificacao else 'green')
        
        ax1.set_title(f'Frame {idx+1}\n{classificacao}\nConfianca: {confianca:.1f}%',
                      fontsize=11, fontweight='bold', color=cor)
        ax1.axis('off')
        
        ax2 = plt.subplot(n_frames, 3, idx*3 + 2)
        im = ax2.imshow(resultado['magnitude'], cmap='hot')
        ax2.set_title(f'Espectro Frame {idx+1}', fontsize=11, fontweight='bold', color='white')
        ax2.set_facecolor('#1a1a1a')
        ax2.axis('off')
        
        ax3 = plt.subplot(n_frames, 3, idx*3 + 3)
        ax3.plot(resultado['caracteristicas']['energia_radial'], linewidth=2, color='#00ff00')
        ax3.fill_between(range(len(resultado['caracteristicas']['energia_radial'])),
                         resultado['caracteristicas']['energia_radial'], alpha=0.3, color='#00ff00')
        ax3.set_title(f'Energia Radial Frame {idx+1}', fontsize=11, fontweight='bold', color='white')
        ax3.set_xlabel('Raio', color='white')
        ax3.set_ylabel('Energia', color='white')
        ax3.set_facecolor('#1a1a1a')
        ax3.grid(True, alpha=0.3, color='#444444')
        ax3.tick_params(colors='white')
    
    plt.tight_layout()
    print("Feche a janela do gráfico para continuar...")
    plt.show()

def gerar_relatorio(resultados):
    if not resultados: return
    print("\n" + "="*80)
    print("RELATORIO DE ANALISE - DETECCAO DE DEEPFAKES".center(80))
    print("="*80)
    print(f"\nTotal de frames analisados: {len(resultados)}".center(80))
    
    sinteticas = sum(1 for r in resultados if 'SINTETICA' in r['classificacao'])
    naturais = sum(1 for r in resultados if 'NATURAL' in r['classificacao'])
    indeterminadas = sum(1 for r in resultados if 'INDETERMINADO' in r['classificacao'])
    
    print(f"\nNaturais: {naturais}".center(80))
    print(f"Indeterminadas: {indeterminadas}".center(80))
    print(f"Sinteticas: {sinteticas}".center(80))
    
    if sinteticas > 0:
        print(f"\nALERTA: {sinteticas} frame(s) podem ser deepfakes!".center(80))
    else:
        print(f"\nNenhum deepfake detectado!".center(80))
    
    print("\n" + "="*80)
    print("DETALHES POR FRAME".center(80))
    print("="*80)
    
    for i, resultado in enumerate(resultados):
        print(f"\nFRAME {i+1}".center(80))
        print(f"Classificacao: {resultado['classificacao']}".center(80))
        print(f"Confianca: {resultado['confianca']:.1f}%".center(80))
        print(f"Score: {resultado['score']}/6".center(80))

# --- FUNÇÕES PRINCIPAIS MODIFICADAS ---

def main_webcam_local():
    print("="*80)
    print("DETECCAO DE DEEPFAKES - VERSAO LOCAL (VS CODE)".center(80))
    print("="*80)
    
    num_frames = 3
    print(f"Vou capturar {num_frames} frames da sua webcam.\n".center(80))
    
    frames = capturar_multiplos_frames_local(num_frames)
    
    if not frames:
        print("Nenhum frame capturado. Abortando.")
        return [], []

    print("\nAnalisando frames...\n".center(80))
    resultados = []
    
    for i, frame in enumerate(frames):
        print(f"Analisando frame {i+1}/{num_frames}...".center(80))
        resultado = classificar_imagem(frame)
        resultados.append(resultado)
    
    print("\nAnalise completa!\n".center(80))
    print("Gerando visualizacoes... (Uma nova janela será aberta)".center(80))
    
    visualizar_multiplos_frames(frames, resultados)
    gerar_relatorio(resultados)
    
    return frames, resultados

def analisar_imagem_local():
    print("\n" + "="*80)
    print("ANALISE DE IMAGEM (MODO MANUAL)".center(80))
    print("="*80)
    
    # Instruções claras para o usuário
    print("\nCOMO PEGAR O CAMINHO DA IMAGEM:")
    print("1. Vá na pasta da imagem.")
    print("2. Segure a tecla SHIFT e clique com o botão DIREITO na imagem.")
    print("3. Escolha 'Copiar como caminho'.")
    print("4. Volte aqui e aperte Ctrl+V.\n")
    
    # Input direto no terminal (sem janela travada)
    filename = input(">>> Cole o caminho aqui e dê Enter: ").strip()
    
    # Limpeza de aspas (o Windows costuma colocar aspas extras)
    filename = filename.strip('"').strip("'")

    # Verificação básica
    if not filename:
        print("\nNenhum caminho digitado. Voltando ao menu.")
        return

    if not os.path.exists(filename):
        print(f"\nERRO: O arquivo não foi encontrado: {filename}")
        print("Verifique se o caminho está correto.")
        return

    print(f"\nCarregando: {filename} ...".center(80))
    
    # Leitura da imagem
    imagem = cv2.imread(filename)
    
    if imagem is None:
        # Tentativa secundária com Pillow (para caminhos com acentos ou formatos diferentes)
        try:
            pil_img = Image.open(filename)
            imagem = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"\nErro fatal ao abrir a imagem: {e}")
            print("Tente mover a imagem para uma pasta simples (ex: C:\\teste.jpg)")
            return
    else:
        # Se abriu com OpenCV, está em BGR (padrão do cv2)
        pass 
    
    # Converter para RGB antes de analisar (o matplotlib espera RGB)
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    # Processamento
    print("Analisando espectro de Fourier...".center(80))
    resultado = classificar_imagem(imagem_rgb)
    
    # Exibição
    print("\nAbrindo visualização gráfica...".center(80))
    visualizar_analise_individual(resultado)
    
    # Relatório no terminal
    print("\n" + "="*80)
    print(f"RESULTADO: {resultado['classificacao']}".center(80))
    print(f"Confianca: {resultado['confianca']:.1f}%".center(80))
    print("="*80)
    print("Detalhes técnicos:")
    for detalhe in resultado['detalhes']:
        print(f" - {detalhe}")
    print("="*80)

def menu_principal():
    while True:
        print("\n" + "="*80)
        print("DETECTOR DE DEEPFAKES - MENU PRINCIPAL (LOCAL)".center(80))
        print("="*80)
        print("\nEscolha uma opcao:\n".center(80))
        print("1 - Capturar frames da WEBCAM".center(80))
        print("2 - Analisar ARQUIVO de imagem".center(80))
        print("3 - Sair\n".center(80))
        
        opcao = input("Digite a opcao (1, 2 ou 3): ".center(80)).strip()
        
        if opcao == "1":
            main_webcam_local()
        elif opcao == "2":
            analisar_imagem_local()
        elif opcao == "3":
            print("\nEncerrando...".center(80))
            break
        else:
            print("\nOpcao invalida!".center(80))

if __name__ == "__main__":

    menu_principal()
