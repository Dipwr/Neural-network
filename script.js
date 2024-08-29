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
    this.weights = weights;
    this.biases = biases;
    this.layers = [];
    this.weightedSums = []

    this.output = [];
    
    this.activationDevs = [];
    this.weightDevs = []; 
    this.biasDevs = [];

    this.actFuncDecider = function(x, isDerivative){
        if(isDerivative){
            //return ((x < this.numLayers-1) ? leakyRelUDev : sigmoidDev);
            return leakyRelUDev;
        }else{
            //return ((x < this.numLayers-1) ? leakyRelU : sigmoid);
            return leakyRelU;
        }
    }

    this.getNeuronsInLayer = function(layer){
        return this.layersSetup[layer];
    }

    this.init = function(){
        for (let i = 0; i < this.numLayers; i++){
            this.layers[i] = new Layer(i, this, this.actFuncDecider(i, false)); //if last layer pass sigmoid if not pass leakyReLU
        }
    }

    this.randWeightsAndBiases = function(){
        this.weights = [];
        this.biases = [];
        for (let i = 0; i < this.numLayers-1; i++){
            this.weights[i] = [];
            this.biases[i] = [];
            this.activationDevs[i] = [];
            this.weightDevs[i] = []; 
            this.biasDevs[i] = [];
            for (let j = 0; j < this.layersSetup[i+1]; j++){
                this.weights[i][j] = [];
                this.weightDevs[i][j] = [];
                this.biases[i][j] = [(Math.random())];
                for (let n = 0; n < this.layersSetup[i]; n++){
                    this.weights[i][j][n] = -1 + Math.random()*2;
                }
            }
        }
    }

    this.randInput = function(){
        this.layers[0].activations = [];
        for (let i = 0; i < this.layersSetup[0]; i++){
            this.layers[0].activations[i] = [(-1 + Math.random()*2)]
        }   
    }

    this.getOutput = function(){
        this.output = this.layers[this.numLayers - 1].calcActivations();
        return this.output;
    }

    this.getCost= function(expectedOutputs){
        let acc = 0;
        for (let i = 0; i < this.numOutputs; i++){
            acc += (this.output[i][0] - expectedOutputs[i][0]) ** 2;
        }
        return acc;
    }

    this.partialDevOfOutputs = function(expectedOutputs){
        let dev = math.multiply(2, math.subtract(this.output, expectedOutputs));
        this.activationDevs[this.numLayers - 1] = dev;
        return dev;
    }

    this.partialDevOfActivation = function(layer, index){
        let acc = 0;
        for (let k = 0; k < this.getNeuronsInLayer(layer + 1); k++){
            let actFuncDevFunc = this.actFuncDecider(layer + 1, true);

            let weight = this.weights[layer][k][index]; //layer+1 not needed as input layer does not have weights
            let actFuncDev = actFuncDevFunc(this.weightedSums[layer + 1][k]);
            let nextLayerActDev = this.activationDevs[layer+1][k][0]; //[0] needed as activationDevs contains vertical matrices

            acc += (weight * actFuncDev * nextLayerActDev);
        }
        this.activationDevs[layer][index] = [acc];
        return [acc];
    }

    this.partialDevOfWeight = function(layer, j, k){
        let actFuncDevFunc = this.actFuncDecider(layer, true);

        let prevAct = this.layers[layer - 1].activations[k][0]; //[0] needed as activations contains a vertical matrix
        let actFuncDev = actFuncDevFunc(this.weightedSums[layer][j]);
        let nextLayerActDev = this.activationDevs[layer][j][0]; //[0] needed as activationDevs contains vertical matrices

        this.weightDevs[layer-1][j][k] = (prevAct * actFuncDev * nextLayerActDev); //layer-1 as the input layer does not have weightDevs
        return this.weightDevs[layer-1][j][k];
    }

    this.partialDevOfBias = function(layer, j){
        let actFuncDevFunc = this.actFuncDecider(layer, true);

        let actFuncDev = actFuncDevFunc(this.weightedSums[layer][j]);
        let nextLayerActDev = this.activationDevs[layer][j][0]; //[0] needed as activationDevs contains vertical matrices

        this.biasDevs[layer-1][j] = [(actFuncDev * nextLayerActDev)]; //layer-1 as the input layer does not have biasDevs
        return this.biasDevs[layer-1][j];
    }

    this.backpropagate = function(expectedOutputs){
        //HAS to be run from outputs to inputs
        this.partialDevOfOutputs(expectedOutputs)

        for (let layer = this.numLayers-1; layer > 0; layer--){
            for (let j = 0; j < this.layersSetup[layer]; j++){
                if (layer != this.numLayers-1){
                    this.partialDevOfActivation(layer, j)
                }

                this.partialDevOfBias(layer, j);

                for (let k = 0; k < this.layersSetup[layer-1]; k++){
                    this.partialDevOfWeight(layer, j, k);
                }
            }
        }
    }

    this.applyNegGadient = function(){
        for (let layer = 0; layer < this.numLayers-1; layer++){
            this.weights[layer] = math.add(this.weights[layer], math.multiply(this.weightDevs[layer], -1));
            this.biases[layer] = math.add(this.biases[layer], math.multiply(this.biasDevs[layer], -1));
        }
    }

    this.resetActivations = function(){
        for (let i = this.numLayers-1; i > 0; i--){ //does not reset inputs
            this.layers[i].resetActivations();
        }
        this.output = [];
    }

    this.descendGradiant = function(func, steps, numData, randomizeWeights){
        if(randomizeWeights){this.randWeightsAndBiases()}
        
        for (let i = 0; i < steps; i++){
            let allGraWeights;
            let allGraBiases;
            let tCost = 0;
            for (let j = 0; j < numData; j++){
                this.randInput()

                let expectedOutputs = func(this.layers[0].activations);

                this.getOutput();

                let cost =  this.getCost(expectedOutputs);
                tCost += cost;
                //console.log("data point: ", j," of step: ",i, ", cost: ", cost, ", outputs: ", this.output);

                this.backpropagate(expectedOutputs);

                if (j == 0){
                    allGraWeights = math.clone(this.weightDevs);
                    allGraBiases = math.clone(this.biasDevs);
                }else{
                    for (let n = 0; n < this.numLayers-1; n++){
                        allGraWeights[n] = math.add(allGraWeights[n], this.weightDevs[n]);
                        allGraBiases[n] = math.add(allGraBiases[n], this.biasDevs[n]);
                    }
                }
                
                this.resetActivations();
            }
            for (let n = 0; n < this.numLayers-1; n++){
                this.weightDevs[n] = math.divide(allGraWeights[n], numData);
                this.biasDevs[n] = math.divide(allGraBiases[n], numData)
            }
            console.log("Average cost of step: ", tCost/numData)
            this.applyNegGadient()
        }
    }
}

function sigmoid(x){
    return 1 / (1 + math.exp(-x))
}

function sigmoidDev(x){
    return sigmoid(x) * (1 - sigmoid(x));
}

function leakyRelU(x){
    if (x > 0) {
        return x;
    }else{
        return 0.01 * x;
    }
}

function leakyRelUDev(x){
    if (x > 0) {
        return 1;
    }else{
        return 0.01;
    }
}

function Layer(index, network, actFunc){
    //REMEMBER to add diffrent object for input layer
    this.index = index;
    this.activationFunction = actFunc;
    this.activations = true;
    
    this.calcActivations = function(){
        if(this.activations != true){return this.activations;}

        let prevActivations = network.layers[this.index - 1].calcActivations();

        let weightedSum = math.add(math.multiply(network.weights[this.index - 1], prevActivations), network.biases[this.index - 1]);//reason for -1 is input layer does not have weights or biases 
        network.weightedSums[index] = weightedSum;

        this.activations = math.map(weightedSum, this.activationFunction);
        return this.activations;
    }

    this.resetActivations = function(){
        this.activations = true;
    }
}

function funcToAprox(input){
    let x = input[0][0]
    let output = Math.sin(Math.PI * x);
    return [[output]]
}

let n = new NeuralNetwork([1,10,10,10,10,1]);
n.init();
console.log(n);
