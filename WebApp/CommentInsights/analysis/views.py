from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def homepage(request):
    return render(request, 'homepage.html')

def upload_file(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    elif request.method == 'POST':
        doc = request.FILES.get("file")

        print(type(doc))

        with open('./static/data/for_prediction.csv', 'wb') as save_file:
            for part in doc.chunks():
                save_file.write(part)
                save_file.flush()

            return render(request, 'upload_result.html')