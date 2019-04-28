//When file input is changed, read the file
$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        //Display the image back on the page
        $("#selected-image").attr("src", dataURL);
        //Remove previous predictions
        $('#prediction-list').hide();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

//Create a new model
let model;

//Load the model and hide the progress bar
(async function() {
    console.log('Loading the model...');
    model = await tf.loadModel(`http://localhost:81/tfjs-models/MobileNet/model.json`);
    console.log('Model Loaded.');
    $('.progress-bar').hide();
})();

$("#predict-button").click(async function() {
    //Get the image
    let image = $('#selected-image').get(0);
    //Transform image to match the model.
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224,224])//Resize the image
        .toFloat()
        //.expandDims();//Change dimensions of tensor to 4D

    let offset = tf.scalar(127.5); //Scale RGB values 255/2

    let processedTensor = tensor.sub(offset) //Original Tensor - Offset: 224-127.5 = 96.5
        .div(offset) //Divide subtracted tensor by offset: 96.5 / 127.5 = 0.7568
        .expandDims(); //Change all values in Tensor on a scale from 0 to 1

    let predictions = await model.predict(processedTensor).data(); //1000 elements
    let top5 = Array.from(predictions)
        .map(function(p, i) { //Map the predictions to the name from file -> Return object
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function(a, b) { //Sort from highest prediction value to lowest
            return b.probability - a.probability;
        }).slice(0,5); //Keep only the top 5 probabilities

    $("#prediction-list").empty();
    $('#prediction-list').append("<th width='33%'>Name</th><th width='33%'>Value</th><th width='33%'>Percentage</th>");
    top5.forEach(function(p) {
        $('#prediction-list').append("<tr><td>" + p.className + "</td><td>" + p.probability.toFixed(6) + "</td><td>" + (p.probability*100).toFixed(2) + "%</td></tr>");
        $("#prediction-list").show();
    });
});

