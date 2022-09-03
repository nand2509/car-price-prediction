from flask import Flask,render_template,request
app = Flask(_name_)
@app.route("/") # this will direct  us to the home page when we click our web app link
def home():
      return render_template("index.html")  # home page
@app.route("/predict", methods = ["POST"]) # this works when the user click the prediction button
def predict():
      year = int(request.form["year"]) # taking year input from the user
      tot_year = 2020 - year
      present_price = float(request.form["present_price"]) #taking the present prize
      fuel_type = request.form["fuel_type"] # type of fuel of car
      # if loop for assigning numerical values
      if fuel_type == "Petrol":
            fuel_P = 1
            fuel_D = 0
      else:
            fuel_P = 0
            fuel_D = 1
      kms_driven = int(request.form["kms_driven"]) # total driven kilometers of the car
      transmission = request.form["transmission"] # transmission type
      # assigning numerical values
      if Transmission == "Manuel":
            transmission_manual = 1
      else:
            transmission_manual = 0
      seller_type = request.form["seller_type"] # seller type
      if seller_type == "Individual":
             seller_individual = 1
      else:
             seller_individual = 0
      owner = int(request.form["owner"])  # number of owners
      values = [[
        present_price,
        kms_driven,
        owner,
        tot_year,
        fuel_D,
        fuel_P,
        seller_individual,
        transmission_manual
      ]]
      # created a list of all the user inputed values, then using it for prediction
      prediction = model.predict(values)
      prediction = round(prediction[0],2)
      # returning the predicted value inorder to display in the front end web application
      return render_template("home.html", pred = "Car price is {} Lakh".format(float(prediction)))
if _name_ == "_main_":
     app.run(debug = True)