import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.fft

print("=== EXPERIMENTO 2: Camada de Rede Neural (Conv2d vs FFT) ===")

# Configuração para usar GPU se disponível (acelera muito a FFT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {device}")

# --- 1. Definição da Camada baseada em FFT ---
class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # Inicializa pesos aleatórios como uma camada normal
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # Para multiplicar na frequência, o kernel e a imagem devem ter o mesmo tamanho.
        # Fazemos padding no kernel para ficar do tamanho da imagem.
        _, _, h, w = x.shape

        # 1. FFT da Entrada (Imagem)
        x_fft = torch.fft.rfft2(x)

        # 2. FFT do Peso (Kernel) - com padding para (h, w)
        w_fft = torch.fft.rfft2(self.weight, s=(h, w))

        # 3. Multiplicação no domínio da frequência (Teorema da Convolução)
        out_fft = x_fft * w_fft

        # 4. IFFT (Volta para o espaço)
        out = torch.fft.irfft2(out_fft, s=(h, w))

        # Ajuste de corte (crop) para manter tamanho original (simplificado)
        return out[:, :, :h, :w]

# --- 2. Configuração do Teste ---
# FFT vence em Kernels GRANDES. Vamos simular uma camada que olha "o todo".
batch_size = 16
canais = 1
tamanho_img = 512
tamanho_kernel = 64 # Kernel grande (64x64) favorece FFT

# Criando dados aleatórios (Batch, Canais, Altura, Largura)
entrada = torch.randn(batch_size, canais, tamanho_img, tamanho_img).to(device)

# --- 3. Teste da Camada Padrão (Conv2d) ---
camada_normal = nn.Conv2d(canais, canais, tamanho_kernel, padding='same').to(device)

print(f"\n1. Testando nn.Conv2d Padrão (Kernel {tamanho_kernel}x{tamanho_kernel})...")
inicio = time.time()
for _ in range(10): # Rodar 10 vezes para média
    _ = camada_normal(entrada)
if device.type == 'cuda': torch.cuda.synchronize() # Esperar GPU terminar
fim = time.time()
tempo_normal = (fim - inicio) / 10
print(f"Tempo médio (Padrão): {tempo_normal:.4f} s")

# --- 4. Teste da Camada FFT ---
camada_fft = FFTConv2d(canais, canais, tamanho_kernel).to(device)

print(f"\n2. Testando FFTConv2d Personalizada (Kernel {tamanho_kernel}x{tamanho_kernel})...")
# Aquecimento (Warmup) para preparar as caches da FFT
_ = camada_fft(entrada)

inicio = time.time()
for _ in range(10):
    _ = camada_fft(entrada)
if device.type == 'cuda': torch.cuda.synchronize()
fim = time.time()
tempo_fft_nn = (fim - inicio) / 10
print(f"Tempo médio (FFT):    {tempo_fft_nn:.4f} s")

# --- 5. Conclusão ---
speedup = tempo_normal / tempo_fft_nn
print(f"\nCONCLUSÃO NA REDE NEURAL:")
if speedup > 1:
    print(f"A camada via FFT foi {speedup:.2f}x mais rápida.")
else:
    print(f"A camada Padrão foi mais rápida. (Isso ocorre em Kernels pequenos ou otimizações Winograd)")
    print("Para ver a FFT vencer, tente aumentar o tamanho_kernel para 100+ ou a resolução da imagem.")

import torch
import torch.nn as nn
import time
import torch.fft
import matplotlib.pyplot as plt
import numpy as np

print("=== EXPERIMENTO 3: Benchmark Conv2d vs FFT (Variação de Kernel) ===")

# Configuração de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {device}")
if device.type == 'cpu':
    print("AVISO: Rodando em CPU. Imagens grandes (512x512) podem demorar na Convolução Padrão.")

# --- 1. Classe FFT (Mantida igual) ---
class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        _, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x)
        w_fft = torch.fft.rfft2(self.weight, s=(h, w))
        out_fft = x_fft * w_fft
        out = torch.fft.irfft2(out_fft, s=(h, w))
        return out[:, :, :h, :w]

# --- 2. Configuração do Loop de Teste ---
# Se seu PC for lento, reduza o último número (ex: tire o 100)
tamanhos_kernel = [5, 10, 20, 40, 60, 64]
tempos_normal = []
tempos_fft = []

batch_size = 32 # Reduzi um pouco o batch para não estourar memória
canais = 1
tamanho_img = 512 

# Dados de entrada fixos
entrada = torch.randn(batch_size, canais, tamanho_img, tamanho_img).to(device)

print(f"\nIniciando testes com imagem {tamanho_img}x{tamanho_img}...")

for k in tamanhos_kernel:
    print(f"Testando Kernel {k}x{k}...", end=" ")
    
    # --- Teste Padrão ---
    conv_normal = nn.Conv2d(canais, canais, k, padding='same').to(device)
    
    # Aquecimento
    _ = conv_normal(entrada)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    # Reduzi para 3 repetições para ser mais rápido e evitar o "travamento"
    inicio = time.time()
    for _ in range(3): 
        _ = conv_normal(entrada)
    if device.type == 'cuda': torch.cuda.synchronize()
    fim = time.time()
    media_normal = (fim - inicio) / 3
    tempos_normal.append(media_normal)
    
    # --- Teste FFT ---
    conv_fft = FFTConv2d(canais, canais, k).to(device)
    
    # Aquecimento
    _ = conv_fft(entrada)
    if device.type == 'cuda': torch.cuda.synchronize()
    
    inicio = time.time()
    for _ in range(3):
        _ = conv_fft(entrada)
    if device.type == 'cuda': torch.cuda.synchronize()
    fim = time.time()
    media_fft = (fim - inicio) / 3
    tempos_fft.append(media_fft)
    
    print(f"Ok. (Normal: {media_normal:.4f}s | FFT: {media_fft:.4f}s)")

# --- 3. Geração do Gráfico ---
plt.figure(figsize=(10, 6))

plt.plot(tamanhos_kernel, tempos_normal, marker='o', label='Convolução Padrão (Espacial)', color='red', linewidth=2)
plt.plot(tamanhos_kernel, tempos_fft, marker='s', label='Convolução via FFT (Frequência)', color='green', linewidth=2)

plt.title('Comparativo de Desempenho: Convolução Espacial vs. FFT\n(Em função do tamanho do Kernel)', fontsize=14)
plt.xlabel('Tamanho do Kernel (pixels)', fontsize=12)
plt.ylabel('Tempo de Execução (segundos)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# --- AJUSTE DA POSIÇÃO DO TEXTO ---
# O xytext controla onde o texto fica. 
# (-40, -30) significa: 40 pontos pra esquerda e 30 pontos pra BAIXO do ponto final
plt.annotate('Crescimento Exponencial\n(Lento)', 
             xy=(tamanhos_kernel[-1], tempos_normal[-1]), 
             xytext=(-10, -30), # <--- Mudei aqui para descer o texto
             textcoords='offset points',
             )

plt.annotate('Custo Constante\n(Rápido)', 
             xy=(tamanhos_kernel[-1], tempos_fft[-1]), 
             xytext=(-40, 20), # <--- Texto da FFT um pouco pra cima para não bater no eixo
             textcoords='offset points',
)

plt.tight_layout()
plt.show()
