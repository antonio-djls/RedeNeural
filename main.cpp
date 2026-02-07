#include <cstdint>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

/* Erro = Resposta Esperada - Resposta prevista
 * Os pesos devem ser calculados até achar um erro de valor mínimo
 *
 */
/* Descrição do Projeto
 *
 *
 *  No código é implementada uma Rede Neural MNIST para detecção de dígitos
 * manuscritos Camada de entrada = 28*28 = 784 neurônios
 *
 *
 *
 *
 */

struct neuronio_entrada {
  double valor{};
  double bias = 1.0;
  std::vector<double> pesos;

  neuronio_entrada() : pesos(128) {}
};

struct neuronio_camadaoculta_dois {
  double valor{};
  double bias = 1.0;
  std::vector<double> pesos;

  neuronio_camadaoculta_dois() : pesos(64) {}
};

struct neuronio_camadaoculta_tres {
  double valor{};
  double bias = 1.0;
  std::vector<double> pesos;

  neuronio_camadaoculta_tres() : pesos(10) {}
};

struct neuronio_saida {
  double valor{};
};

class rede {
public:
  std::vector<neuronio_entrada> primeira_camada;
  std::vector<neuronio_camadaoculta_dois> segunda_camada;
  std::vector<neuronio_camadaoculta_tres> terceira_camada;
  std::vector<neuronio_saida> ultima_camada;

  rede() {
    primeira_camada.resize(784);
    segunda_camada.resize(128);
    terceira_camada.resize(64);
    ultima_camada.resize(10);
  }
  //  Chamar diversas vezes para treinamento da rede neural
  rede feed_forward(rede &rede_neural) {

    for (int k = 0; k < 128; k++) {
      double sum{};
      for (int i = 0; i < 784; i++) {
        sum += primeira_camada[i].valor * primeira_camada[i].pesos[k];
      }
      sum += primeira_camada[0].bias * 0.1; // bias term
      segunda_camada[k].valor = sigmoid(sum);
    }

    for (int k = 0; k < 64; k++) {
      double sum{};
      for (int i = 0; i < 128; i++) {
        sum += segunda_camada[i].valor * segunda_camada[i].pesos[k];
      }
      sum += segunda_camada[0].bias * 0.1; // bias term
      terceira_camada[k].valor = sigmoid(sum);
    }
    for (int k = 0; k < 10; k++) {
      double sum{};
      for (int i = 0; i < 64; i++) {
        sum += terceira_camada[i].valor * terceira_camada[i].pesos[k];
      }
      sum += terceira_camada[0].bias * 0.1; // bias term
      ultima_camada[k].valor = sigmoid(sum);
    }
    return rede_neural;
  }

  // Retro propagação
  void back_forward(const std::vector<double> &target_output,
                    double learning_rate = 0.01) {
    // Calcular erro na camada de saída
    std::vector<double> output_error(10);
    std::vector<double> output_delta(10);

    for (int i = 0; i < 10; i++) {
      output_error[i] = target_output[i] - ultima_camada[i].valor;
      output_delta[i] = output_error[i] * ultima_camada[i].valor *
                        (1.0 - ultima_camada[i].valor);
    }

    // Calcular erro e delta na terceira camada (oculta 2)
    std::vector<double> hidden3_error(64);
    std::vector<double> hidden3_delta(64);

    for (int i = 0; i < 64; i++) {
      hidden3_error[i] = 0.0;
      for (int j = 0; j < 10; j++) {
        hidden3_error[i] += output_delta[j] * terceira_camada[i].pesos[j];
      }
      hidden3_delta[i] = hidden3_error[i] * terceira_camada[i].valor *
                         (1.0 - terceira_camada[i].valor);
    }

    // Calcular erro e delta na segunda camada (oculta 1)
    std::vector<double> hidden2_error(128);
    std::vector<double> hidden2_delta(128);

    for (int i = 0; i < 128; i++) {
      hidden2_error[i] = 0.0;
      for (int j = 0; j < 64; j++) {
        hidden2_error[i] += hidden3_delta[j] * segunda_camada[i].pesos[j];
      }
      hidden2_delta[i] = hidden2_error[i] * segunda_camada[i].valor *
                         (1.0 - segunda_camada[i].valor);
    }

    // Atualizar pesos da terceira camada para saída
    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < 10; j++) {
        terceira_camada[i].pesos[j] +=
            learning_rate * output_delta[j] * terceira_camada[i].valor;
      }
    }

    // Atualizar pesos da segunda camada para terceira
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 64; j++) {
        segunda_camada[i].pesos[j] +=
            learning_rate * hidden3_delta[j] * segunda_camada[i].valor;
      }
    }

    // Atualizar pesos da entrada para segunda camada
    for (int i = 0; i < 784; i++) {
      for (int j = 0; j < 128; j++) {
        primeira_camada[i].pesos[j] +=
            learning_rate * hidden2_delta[j] * primeira_camada[i].valor;
      }
    }
  }

  // Função de Ativação do Neurônio
  double sigmoid(double x) { return 1.0 / (1 + std::exp(-x)); }

  // Função para converter label para one-hot encoding
  std::vector<double> label_to_one_hot(uint8_t label) {
    std::vector<double> one_hot(10, 0.0);
    one_hot[label] = 1.0;
    return one_hot;
  }

  // Função para obter predição (índice do maior valor de saída)
  int get_prediction() {
    int max_index = 0;
    double max_value = ultima_camada[0].valor;

    for (int i = 1; i < 10; i++) {
      if (ultima_camada[i].valor > max_value) {
        max_value = ultima_camada[i].valor;
        max_index = i;
      }
    }
    return max_index;
  }
};

struct MNIST_Image {
  uint32_t magic_number;
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_columns;
  std::vector<uint8_t> pixels;
};
struct MNIST_Label {
  uint32_t magic_number;
  uint32_t num_items;
  std::vector<uint8_t> labels;
};

uint32_t read_big_endian(std::ifstream &file) {
  uint32_t value;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));
  return __builtin_bswap32(value);
}

bool load_mnist_images(const std::string &filename, MNIST_Image &mnist) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Erro ao abrir arquivo: " << filename << std::endl;
    return false;
  }

  mnist.magic_number = read_big_endian(file);
  mnist.num_images = read_big_endian(file);
  mnist.num_rows = read_big_endian(file);
  mnist.num_columns = read_big_endian(file);

  mnist.pixels.resize(mnist.num_images * mnist.num_rows * mnist.num_columns);
  file.read(reinterpret_cast<char *>(mnist.pixels.data()), mnist.pixels.size());

  file.close();
  return true;
}

bool load_mnist_labels(const std::string &filename, MNIST_Label &mnist) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Erro ao abrir arquivo: " << filename << std::endl;
    return false;
  }

  mnist.magic_number = read_big_endian(file);
  mnist.num_items = read_big_endian(file);

  mnist.labels.resize(mnist.num_items);
  file.read(reinterpret_cast<char *>(mnist.labels.data()), mnist.labels.size());

  file.close();
  return true;
}

int main() {
  std::cout << "Iniciando a rede neural \n";

  MNIST_Image train_images, test_images;
  MNIST_Label train_labels, test_labels;

  // Carregar dados de treinamento
  if (!load_mnist_images("./mnist/train-images.idx3-ubyte", train_images)) {
    std::cerr << "Falha ao carregar imagens de treinamento" << std::endl;
    return 1;
  }

  if (!load_mnist_labels("./mnist/train-labels.idx1-ubyte", train_labels)) {
    std::cerr << "Falha ao carregar labels de treinamento" << std::endl;
    return 1;
  }

  // Carregar dados de teste
  if (!load_mnist_images("./mnist/t10k-images.idx3-ubyte", test_images)) {
    std::cerr << "Falha ao carregar imagens de teste" << std::endl;
    return 1;
  }
  if (!load_mnist_labels("./mnist/t10k-labels.idx1-ubyte", test_labels)) {
    std::cerr << "Falha ao carregar labels de teste" << std::endl;
    return 1;
  }
  std::cout << "Dataset MNIST carregado com sucesso!" << std::endl;
  std::cout << "Imagens de treinamento: " << train_images.num_images
            << std::endl;
  std::cout << "Labels de treinamento: " << train_labels.num_items << std::endl;
  std::cout << "Imagens de teste: " << test_images.num_images << std::endl;
  std::cout << "Labels de teste: " << test_labels.num_items << std::endl;
  rede rede_um;

  // Extrair primeira imagem de treinamento (784 pixels = 28x28)
  std::cout << "\n=== Primeira Imagem ===\n";
  std::cout << "Entradas (784 pixels da primeira imagem):\n";

  // Converter pixels uint8_t para double (normalizado de 0-255 para 0-1)
  // Preciso fazer essa regularização para lidar com os tipos diferentes
  for (int k = 0; k < 100; k++) {
    std::vector<double> entradas(784);
    for (int i = 0; i < 784; i++) {
      entradas[i] = static_cast<double>(train_images.pixels[i]) / 255.0;
    }

    // Inserindo os valores na camada inicial da Rede Neural
    for (int k = 0; k < 784; k++) {
      rede_um.primeira_camada[k].valor = entradas[k];
    }
  }
  // Mostrar valor esperado (label)
  uint8_t label_esperado = train_labels.labels[0];
  std::cout << "Valor esperado (label): " << static_cast<int>(label_esperado)
            << std::endl;

  // Treinamento da rede neural
  std::cout << "\n=== Iniciando Treinamento ===\n";
  const int epochs = 100;
  const int batch_size = 1000;
  const double learning_rate = 0.01;

  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_error = 0.0;
    int correct_predictions = 0;

    std::cout << "Época " << (epoch + 1) << "/" << epochs << std::endl;

    for (int img = 0; img < batch_size && img < train_images.num_images;
         img++) {
      // Carregar imagem
      for (int i = 0; i < 784; i++) {
        rede_um.primeira_camada[i].valor =
            static_cast<double>(train_images.pixels[img * 784 + i]) / 255.0;
      }

      // Feed forward
      rede_um.feed_forward(rede_um);

      // Preparar target (one-hot encoding)
      std::vector<double> target =
          rede_um.label_to_one_hot(train_labels.labels[img]);

      // Calcular erro
      for (int i = 0; i < 10; i++) {
        double error = target[i] - rede_um.ultima_camada[i].valor;
        total_error += error * error;
      }

      // Verificar predição
      int prediction = rede_um.get_prediction();
      if (prediction == train_labels.labels[img]) {
        correct_predictions++;
      }
      // Backpropagation
      rede_um.back_forward(target, learning_rate);
    }

    double accuracy = (double)correct_predictions / batch_size * 100.0;
    double avg_error = total_error / (batch_size * 10);

    std::cout << "  Acurácia: " << accuracy << "%" << std::endl;
    std::cout << "  Erro médio: " << avg_error << std::endl;
  }

  // Teste da rede neural
  std::cout << "\n=== Iniciando Teste ===\n";
  int test_correct = 0;
  const int test_size = 1000;

  for (int img = 0; img < test_size && img < test_images.num_images; img++) {
    // Carregar imagem de teste
    for (int i = 0; i < 784; i++) {
      rede_um.primeira_camada[i].valor =
          static_cast<double>(test_images.pixels[img * 784 + i]) / 255.0;
    }

    // Feed forward
    rede_um.feed_forward(rede_um);

    // Verificar predição
    int prediction = rede_um.get_prediction();
    if (prediction == test_labels.labels[img]) {
      test_correct++;
    }
  }

  double test_accuracy = (double)test_correct / test_size * 100.0;
  std::cout << "Acurácia no teste: " << test_accuracy << "% (" << test_correct
            << "/" << test_size << ")" << std::endl;

  return 0;
}
