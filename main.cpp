#include<iostream>
#include<math.h>
#include<vector>
#include<fstream>
#include<cstdint>


/* Erro = Resposta Esperada - Resposta prevista
 * Os pesos devem ser calculados até achar um erro de valor mínimo
 *
 */


/* Descrição do Projeto
 *
 *
 *  No código é implementada uma Rede Neural MNIST para detecção de dígitos manuscritos
 *  Camada de entrada = 28*28 = 784 neurônios
 *
 *
 *
 *
 */

struct neuronio_entrada{
  double valor{};
  std::vector<double> pesos;

  neuronio_entrada() : pesos(128) {}
};

struct neuronio_camadaoculta_dois{
  double valor{};
  std::vector<double> pesos;

  neuronio_camadaoculta_dois() : pesos(64) {}
};

struct neuronio_camadaoculta_tres{
  double valor{};
  std::vector<double> pesos;

  neuronio_camadaoculta_tres() : pesos(10) {}
};

struct neuronio_saida{
  double valor{};
};


class rede {
  public:
    std::vector<neuronio_entrada> primeira_camada;
    std::vector<neuronio_camadaoculta_dois> segunda_camada;
    std::vector<neuronio_camadaoculta_tres> terceira_camada;
    std::vector<neuronio_saida> ultima_camada;

    rede(){
        primeira_camada.resize(784);
        segunda_camada.resize(128);
        terceira_camada.resize(64);
        ultima_camada.resize(10);
    }



    rede feed_forward(rede & rede_neural){

        for(int k = 0; k < 128; k++){
            double sum{};
            for(int i = 0; i < 784; i++){
                sum += primeira_camada[i].valor * primeira_camada[i].pesos[k];
            }
            segunda_camada[k].valor = sigmoid(sum);
        }

        for(int k = 0; k < 64; k++){
            double sum{};
            for(int i = 0; i < 128; i++){
                sum += segunda_camada[i].valor * segunda_camada[i].pesos[k];
            }
            terceira_camada[k].valor = sigmoid(sum);
        }
        for(int k = 0; k < 64; k++){
            double sum {};
            for(int i = 0; i < 10; i++){
                sum += terceira_camada[i].valor * terceira_camada[i].pesos[k];
            }
            ultima_camada[k].valor = sigmoid(sum);
        }
        return rede_neural;
    }

    // Função de Ativação do Neurônio
  double sigmoid(double x){
    return 1.0/ (1+ std::exp(-x));
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

uint32_t read_big_endian(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return __builtin_bswap32(value);
}

bool load_mnist_images(const std::string& filename, MNIST_Image& mnist) {
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
    file.read(reinterpret_cast<char*>(mnist.pixels.data()), mnist.pixels.size());

    file.close();
    return true;
}

bool load_mnist_labels(const std::string& filename, MNIST_Label& mnist) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir arquivo: " << filename << std::endl;
        return false;
    }

    mnist.magic_number = read_big_endian(file);
    mnist.num_items = read_big_endian(file);

    mnist.labels.resize(mnist.num_items);
    file.read(reinterpret_cast<char*>(mnist.labels.data()), mnist.labels.size());

    file.close();
    return true;
}

int main(){
    std::cout << "Iniciando a rede neural \n";

    MNIST_Image train_images, test_images;
    MNIST_Label train_labels, test_labels;

    //Carregar dados de treinamento
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
    std::cout << "Imagens de treinamento: " << train_images.num_images << std::endl;
    std::cout << "Labels de treinamento: " << train_labels.num_items << std::endl;
    std::cout << "Imagens de teste: " << test_images.num_images << std::endl;
    std::cout << "Labels de teste: " << test_labels.num_items << std::endl;

    rede rede_um;

    // Extrair primeira imagem de treinamento (784 pixels = 28x28)
    std::cout << "\n=== Primeira Imagem ===\n";
    std::cout << "Entradas (784 pixels da primeira imagem):\n";

    // Converter pixels uint8_t para double (normalizado de 0-255 para 0-1)
    // Preciso fazer essa regularização para lidar com os tipos diferentes
    for(int k = 0; k < 100; k++){
        std::vector<double> entradas(784);
        for (int i = 0; i < 784; i++) {
            entradas[i] = static_cast<double>(train_images.pixels[i]) / 255.0;
        }

        // Inserindo os valores na camada inicial da Rede Neural
        for(int k = 0; k < 784; k++){
            rede_um.primeira_camada[k].valor = entradas[k];
        }

    }

    // Mostrar valor esperado (label)
    uint8_t label_esperado = train_labels.labels[0];
    std::cout << "Valor esperado (label): " << static_cast<int>(label_esperado) << std::endl;

    return 0;
}
