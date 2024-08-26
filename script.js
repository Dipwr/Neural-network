
function NeuralNetwork(layers){
    this.layers = layers;
    this.numLayers = layers.length;
    this.numHiddenLayers = this.numLayers - 2;
    this.numInputs = layers[0];
    this.numOutputs = layers[this.numLayers - 1];
    this.neurons = [];

    this.getNeuronsInLayer = function(layer){
        return this.layers[layer];
    }

    this.init = function(){
        for (let i = 0; i < this.numLayers; i++){
            this.neurons[i] = [];
            for (let j = 0; j < this.layers[i]; j++){
                this.neurons[i][j] = new Neuron(i, j, this.neurons); 
            }
        }
    }
}

let n = new NeuralNetwork([3, 2, 3])
n.init();
console.log(n.neurons[0])

function Neuron(layer, pos, neurons){
    this.layer = layer;
    this.position = pos;
    this.activation = NaN;

    this.getActivation = function(){
        if(this.activation == NaN){
            if(this.layer != 0){
                let activations = neurons[layer-1]
            }
        }
    }
}