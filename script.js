/* 
Layout of weights:
[
    [ 
        [1 w00, 1 w01, 1 w02],    (weights of connections between layer 0 and 1)
        [1 w10, 1 w11, 1 w12] 
    ], 

    [ 
        [2 w00, 2 w02], 
        [2 w10, 2 w12],   (weights of connections between layer 1 and 2)
        [2 w20, 2 w22]  
    ]
]
    
Layout of biases:
[
    [
        [1b0],  (biases of layer 1)
        [1b1]
    ],

    [
        [2b0],  (biases of layer 2)
        [2b1],
        [2b2]
    ]
]
*/

function NeuralNetwork(layersSet, weights, biases){
    this.layersSetup = layersSet;
    this.numLayers = this.layersSetup.length;
    this.numHiddenLayers = this.numLayers - 2;
    this.numInputs = this.layersSetup[0];
    this.numOutputs = this.layersSetup[this.numLayers - 1];
    this.weights =  weights;
    this.biases = biases;
    this.layers = [];
    this.test = [1];



    this.getNeuronsInLayer = function(layer){
        return this.layersSetup[layer];
    }

    this.init = function(){
        for (let i = 0; i < this.numLayers; i++){
            this.layers[i] = new Layer(i, this, (i < this.numLayers-1) ? leakyRelU : sigmoid); //if last layer pass sigmoid if not pass leakyReLU
        }
    }

    this.randWeightsAndBiases = function(){
        this.weights = [];
        this.biases = [];
        for (let i = 0; i < this.numLayers-1; i++){
            this.weights[i] = [];
            this.biases[i] = [];
            for (let j = 0; j < this.layersSetup[i+1]; j++){
                this.weights[i][j] = [];
                this.biases[i][j] = [Math.random()*5];
                for (let n = 0; n < this.layersSetup[i]; n++){
                    this.weights[i][j][n] = -1 + Math.random()*2;
                }
            }
        }
    }

    this.randInput = function(){
        this.layers[0].activations = [];
        for (let i = 0; i < this.layersSetup[0]; i++){
            this.layers[0].activations[i] = [Math.random()]
        }   
    }
}

function sigmoid(x){
    return 1 / (1 + math.exp(-x))
}

function leakyRelU(x){
    if (x > 0) {
        return x;
    }else{
        return 0.01 * x;
    }
}

function Layer(pos, network, actFunc,){
    //REMEMBER to add diffrent object for input layer
    this.pos = pos;
    this.activationFunction = actFunc;
    this.activations = "TEST";
    
    this.calcActivations = function(){
        if(this.activations != "TEST"){return this.activations;}

        let prevActivations = network.layers[this.pos - 1].calcActivations();

        let weightedSum = math.add(math.multiply(network.weights[this.pos - 1], prevActivations), network.biases[this.pos - 1]);//reson for -1 is input layer does not have weights or biases 

        this.activations = math.map(weightedSum, this.activationFunction);
        return this.activations;
    }

    this.resetActivations = function(){
        this.activations = "TEST";
    }
}

let n = new NeuralNetwork([1000,100,50,30,20,10,5]);
n.init();
console.log(n);
