#define ll long long 
#include<math.h>
#include<iostream>
#include<vector>
#include<cstdlib>
#include<new>
#include"estruturas.h"

/*Tamanho da Camada Oculta
    2/3 da camada de entrada + Tamanho da camada de saída
    2/3* 2074600 + 10 = 1.382.410 neurônios
  Tamanho da Camada de entrada
  Considerando uma imagem FULL HD (1920x1080) -> 1920*1080 = 2.073.600 neurônios

*/
// double sigmoid()
// double erro()




class rede{
    public:     
    //     ll int camada_de_entrada{2073600};
    //     ll int camada_oculta{1290};
    //     ll int camada_de_saida{10};
        long double  valor_calculado;
        std::vector<neuronio_entrada> vetor_entrada;
        std::vector<neuronio_oculto> vetor_oculto;
        std::vector<neuronio_saida> vetor_saida;
    // std::vector<ll double>
    
    double sigmoid(double x){
        x = 1 / (1 + exp(-1 * x));
        return x;
    }
    

    // Em andamento, calculado o valor depois da função de somatorio e função de ativação
    void ajuste_de_pesos(){
        long double somatorio{};
        for(int u = 0;  u < sizeof(vetor_entrada); u++){
            for(int k = 0; k < size(vetor_entrada); k++){
                somatorio += (vetor_entrada[k].valor * vetor_entrada[k].array[u]);
            }
            valor_calculado = sigmoid(somatorio);
            vetor_oculto[u].valor = valor_calculado;
            somatorio = 0;
        }
        for(int u = 0; u < sizeof(vetor_saida); u++){
            // Camada Oculta tem 1290 neurônios
            
            for(int k = 0; k < sizeof(vetor_oculto); k++){
                somatorio  += vetor_oculto[k].valor * vetor_oculto[k].array[u];
            }
            valor_calculado = sigmoid(somatorio);
            vetor_saida[u].valor = valor_calculado;
            somatorio = 0;
        }
}


};

int main(){
    // srand(time(NULL));
    // aletorio = rand() % 10 +1; doubleervalo entre 1 e 10
    
    rede r1;


    return 0;
}