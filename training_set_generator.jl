module TrainingSetGenerator

#Declare linear function, which separates observations in 2D space
f(x) = 8*x+3.5

#Generating training data with labels for given N lenght
function generate_data(n)
  training_set = []
  for i in 1:n
    x = rand(0:200)
    y = rand(0:200)
    bias = rand(0:200)
    lineY = f(x)
    if y>lineY
      label = -1
    else
      label = 1
    end
    input_values = [x,y, bias]
    push!(training_set, [input_values, label])
  end
  return training_set
end

end
