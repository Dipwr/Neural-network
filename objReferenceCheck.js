function Object1(layers){
    this.neurons = [];
    this.init = function(){
        for (let i = 0; i < layers; i++){
            this.neurons[i] = new Neuron(this.neurons);
        };
    }
}

let obj = new Object1(3);
obj.init();
console.log(obj);

function Neuron(neurons){
    this.activation = 1;

    this.getActivation = function(){
        console.log(neurons);
    }
}