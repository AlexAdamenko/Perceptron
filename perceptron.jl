include("./training_set_generator.jl")
using TrainingSetGenerator

#Perceptron realization class
type Perceptron
    inputs_number::Int8
    weights::Array{Float64}
    learning_rate::Float64
end

#Predicting class based on the observation vector
function predict(observation, perceptron)
    sum = dot(observation[1], perceptron.weights)
    return return_sign(sum)
end

#Signum
function return_sign(number)
    if number > 0
        return 1
    else
        return -1
    end
end

#Method to recalculate weights according to the error
function recalculate_weight(observation, perceptron)
        label = observation[2]
        prediction = predict(observation, perceptron)
        error = label - prediction
        #Recalculate weights
        perceptron.weights += error * observation[1] * perceptron.learning_rate
        return perceptron, prediction, label
end

#Calculating error rate of the current epoch
function error_rate(predictions, targets)
    total_positives = 0
    for i = 1:length(targets)
        if targets[i] == predictions[i]
            total_positives +=1
        end
    end
    accuracy = total_positives/length(targets)
    return accuracy
end

#Method to train perceptron
function train(inputs, perceptron)
    predictions = []
    true_labels = []
    for i=1:length(inputs)
        perceptron, prediction, label = recalculate_weight(inputs[i], perceptron)
        push!(predictions, prediction)
        push!(true_labels, label)
    end
    accuracy = error_rate(predictions, true_labels)
    return perceptron, accuracy
end

#Method to test perceptron
function test(inputs, perceptron)
    testing_predictions =  []
    true_labels = []
    for i=1:length(inputs)
        prediction = predict(inputs[i], perceptron)
        push!(testing_predictions, prediction)
        push!(true_labels, inputs[i][2])
    end
    accuracy = error_rate(testing_predictions, true_labels)
    println(string("Accuracy on the testing data is: ", accuracy * 100, "%"))
    return perceptron
end

#Metod to run perceptron
function run_perceptron(epochs)

    #Initializing perceptron with, three inputs(two coordinates and a bias) with random weights
    inputs_number = 3
    bias = 1
    learning_rate = 0.2
    perceptron = Perceptron(inputs_number, [rand(-1:1), rand(-1:1), bias], learning_rate)
    accuracy = 0

    sample_training_set = TrainingSetGenerator.generate_data(200)

    for i= 1:epochs
        perceptron, accuracy = train(sample_training_set, perceptron)
    end

    println(string("Accuracy on the training data is: ", accuracy * 100, "%"))
    testing_data = TrainingSetGenerator.generate_data(20)
    test(testing_data, perceptron)

end
