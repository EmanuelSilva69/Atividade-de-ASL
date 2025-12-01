
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
    cv2.destroyAllWindows()
    return frames

# --- FUNÇÕES DE PROCESSAMENTO (MANTIDAS IGUAIS) ---

def converter_para_escala_cinza(imagem):
    if len(imagem.shape) == 3:
        if imagem.shape[2] == 3:
            return cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        elif imagem.shape[2] == 4:
            return cv2.cvtColor(imagem, cv2.COLOR_RGBA2GRAY)
    return imagem

def redimensionar_imagem(imagem, height=512, width=512):
    return cv2.resize(imagem, (width, height))

def calcular_espectro_fourier(imagem):
    imagem_float = imagem.astype(np.float32)
    fft_resultado = fft2(imagem_float)
    fft_shifted = fftshift(fft_resultado)
    magnitude = np.log1p(np.abs(fft_shifted))
    return magnitude, fft_shifted

def extrair_caracteristicas_espectrais(magnitude):
    magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2) # Corrigido operador de potência **
    
    raios = np.linspace(0, max(h, w) // 2, 100)
    energia_radial = []
    
    for r in raios:
        mascara = (R >= r - 1) & (R < r + 1)
        if mascara.sum() > 0:
            energia_radial.append(magnitude_norm[mascara].mean())
        else:
            energia_radial.append(0)
    
    energia_radial = np.array(energia_radial)
    
    variancia_angular = []
    for r in raios[10:80]:
        mascara = (R >= r - 2) & (R < r + 2)
        if mascara.sum() > 5:
            valores = magnitude_norm[mascara]
            variancia_angular.append(np.std(valores))
        else:
            variancia_angular.append(0)
    
    variancia_angular = np.array(variancia_angular)
    
    pico_energia = np.max(energia_radial)
    energia_media = np.mean(energia_radial)
    ratio_pico = pico_energia / (energia_media + 1e-6)
    
    alta_freq = magnitude_norm[-50:, -50:].mean()
    baixa_freq = magnitude_norm[50:100, 50:100].mean()
    ratio_freq = alta_freq / (baixa_freq + 1e-6)
    
    caracteristicas = {
        'energia_radial': energia_radial,
        'variancia_angular': variancia_angular,
        'pico_energia': pico_energia,
        'ratio_pico': ratio_pico,
        'ratio_freq_alta_baixa': ratio_freq,
        'energia_media': energia_media,
        'desvio_padrao_espectral': np.std(magnitude_norm)
    }
    
    return caracteristicas

def classificar_imagem(imagem, threshold_ratio_pico=15.0, threshold_ratio_freq=0.15):
    imagem_cinza = converter_para_escala_cinza(imagem)
    imagem_redimensionada = redimensionar_imagem(imagem_cinza)
    
    mag, _ = calcular_espectro_fourier(imagem_redimensionada)
    caracteristicas = extrair_caracteristicas_espectrais(mag)
    
    score = 0
    detalhes = []
    
    if caracteristicas['ratio_pico'] > threshold_ratio_pico:
        score += 3
        detalhes.append(f"Ratio Pico: {caracteristicas['ratio_pico']:.2f}")
    else:
        detalhes.append(f"Ratio Pico: {caracteristicas['ratio_pico']:.2f}")
    
    if caracteristicas['ratio_freq_alta_baixa'] < threshold_ratio_freq:
        score += 2
        detalhes.append(f"Ratio Freq: {caracteristicas['ratio_freq_alta_baixa']:.4f}")
    else:
        detalhes.append(f"Ratio Freq: {caracteristicas['ratio_freq_alta_baixa']:.4f}")
    
    if caracteristicas['desvio_padrao_espectral'] < 0.15:
        score += 1
        detalhes.append(f"Desvio: {caracteristicas['desvio_padrao_espectral']:.4f}")
    else:
        detalhes.append(f"Desvio: {caracteristicas['desvio_padrao_espectral']:.4f}")
    
    if score >= 4:
        classificacao = "SINTETICA"
        cor = (0, 0, 255) # RGB para plot (Azul no original estava assim, mas geralmente sintético é alerta)
        confianca = min(100, (score / 6) * 100)
    elif score >= 2:
        classificacao = "INDETERMINADO"
        cor = (255, 165, 0) # Laranja
        confianca = (score / 6) * 100
    else:
        classificacao = "NATURAL"
        cor = (0, 255, 0) # Verde
        confianca = 100 - (score / 6) * 100
    
    return {
        'classificacao': classificacao,
        'confianca': confianca,
        'score': score,
        'cor': cor,
        'detalhes': detalhes,
        'caracteristicas': caracteristicas,
        'magnitude': mag,
        'imagem_cinza': imagem_redimensionada
    }

# --- FUNÇÕES DE VISUALIZAÇÃO ---

def visualizar_analise_individual(resultado):
    """Substitui visualizar_analise_webcam para funcionar genericamente"""
    # Matplotlib no VSCode abre uma janela nova.
    # Usando plt.ion() ou apenas show() normal.
    
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(resultado['imagem_cinza'], cmap='gray')
    ax1.set_facecolor('#1a1a1a')
    
    classificacao = resultado['classificacao']
    confianca = resultado['confianca']
    cor_texto = 'red' if 'SINTETICA' in classificacao else ('orange' if 'INDETERMINADO' in classificacao else 'green')
    
    ax1.set_title(f'IMAGEM ANALISADA\n{classificacao}\nConfianca: {confianca:.1f}%', 
                  fontsize=12, fontweight='bold', color=cor_texto)
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(resultado['magnitude'], cmap='hot')
    ax2.set_title('ESPECTRO DE FOURIER\nArtefatos em vermelho', fontsize=12, fontweight='bold', color='white')
    ax2.set_facecolor('#1a1a1a')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_facecolor('#1a1a1a')
    
    char = resultado['caracteristicas']
    metricas_text = f"""
ANALISE ESPECTRAL

Classificacao: {resultado['classificacao']}
Confianca: {resultado['confianca']:.1f}%
Score: {resultado['score']}/6

Ratio Pico: {char['ratio_pico']:.2f}

Ratio Freq Alta/Baixa:
{char['ratio_freq_alta_baixa']:.4f}

Desvio Espectral:
{char['desvio_padrao_espectral']:.4f}

Energia Media:
{char['energia_media']:.4f}
"""
    
    for detalhe in resultado['detalhes']:
        metricas_text += f"\n{detalhe}"
    
    ax3.text(0.05, 0.95, metricas_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9, edgecolor='#444444'),
             color='#00ff00')
    
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
    print("ANALISE DE IMAGEM (ARQUIVO LOCAL)".center(80))
    print("="*80)
    
    filename = ""
    if HAS_TKINTER:
        # Abre janela para escolher arquivo
        root = tk.Tk()
        root.withdraw() # Esconde a janela principal do TK
        print("Selecione a imagem na janela que abriu...")
        filename = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        root.destroy()
    else:
        filename = input("Digite o caminho completo da imagem: ").strip().strip('"')

    if not filename or not os.path.exists(filename):
        print("Arquivo não selecionado ou não encontrado.")
        return

    print(f"\nAnalisando: {filename}\n".center(80))
    
    # Lê imagem usando OpenCV (lida bem com caminhos locais)
    imagem = cv2.imread(filename)
    
    if imagem is None:
        # Fallback para PIL se OpenCV falhar (ex: caracteres especiais ou formatos exóticos)
        try:
            imagem = Image.open(filename)
            imagem = np.array(imagem)
        except:
            print("Erro ao ler imagem.")
            return
    else:
        # OpenCV lê em BGR, converter para RGB
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    
    resultado = classificar_imagem(imagem)
    visualizar_analise_individual(resultado)
    
    print("\n" + "="*80)
    print(f"RESULTADO: {resultado['classificacao']}".center(80))
    print(f"Confianca: {resultado['confianca']:.1f}%".center(80))
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
