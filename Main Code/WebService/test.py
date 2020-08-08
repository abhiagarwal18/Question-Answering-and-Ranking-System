bits = 0
    response_data={}
    for token in questionString.split():
        token = token.lower()
        if token == 'bits' or token == 'pilani':
            bits = 1

    if bits == 1:
        predicted_class = "Bits Pilani"
        response['0']=predicted_class
        response['1']="sample answer"
        return jsonify(response_data)