#include <bits/stdc++.h>

using namespace std;








const int MAXN = 6e4;
double image[MAXN][30][30];
unsigned int num, magic, rows, cols;
int currBatch=0;
unsigned int label[MAXN];


unsigned int in(ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}

void input() {
    ifstream icin;
    icin.open("C:/Users/Asus/Desktop/NN/data/train-images-idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                image[i][x][y] = in(icin, 1);
                image[i][x][y]/=(255*1.0);
            }
        }
    }
    icin.close();
    icin.open("C:/Users/Asus/Desktop/NN/data/train-labels-idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        label[i] = in(icin, 1);
    }
}


int main(){
    input();
    for(int i=0;i<3;i++){
        for(int k=0;k<30;k++){
            for(int j=0;j<30;j++){
                cout<<image[i][k][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"NEXT IMAGE"<<endl;
    };
}