INFO

I. CONFIGURATION
1. Open sever/settings.py
    - search for DATABASES settings
    - be sure to have the correct configuration for your DB (name, user, password)

II. USE
1. Start the app
    - in terminal, go to the path where manage.py is
    - python manage.py runserver

2. Open on web
    - open http://127.0.0.1:8000/

3. See the 3 pages. Start the operations from left to right.

4. Open student-activities page
    - here you can see the student's activities that are done and can be analyzed
    STEPS:
        - choose a student from DB
        - choose the 'done' activities
        - choose the activity types you want. It's a LIKE in SQL, comma separated (e.g. 'emotion,delay'), leave empty for all
        - choose the rows count limit, -1 default (all)
        - see the results in the table
        - UNDER THE INPUTS, YOU HAVE THE ACTIVITY IDS AND YOU CAN COPY THEM FOR THE NEXT PAGE.

5. Open next page - Sensor Heart Rate
    - here you can see the table sensor_heart_rate based on activity ids
    STEPS:
        - PASTE the previous copied activity ids
        - set the limit to -1 to take all and click 'Search'
        - click 'Download CSV' to download a CSV with all the data from the table
        - CLICK 'DOWNLOAD PROCESSED CSV' TO DOWNLOAD THE CSV READY FOR ANALYSIS

6. Go to the third page - HRV Metrics
    - here you can upload the processed CSV and compute the HRV metrics.
    - the HRV metrics are computed using 2 methods: pyhrv library and manual written functions
    - there will be 2 tables with results, and you will download the one computed with pyhrv (this can be changed easy in the code).
    -STEPS:
        - UPLOAD THE PREVIOUS GENERATED CSV
        - you can compute and see the results in a table (grouped by activity id) or COMPUTE AND DOWNLOAD A CSV WITH THE RESULTS (pyhrv)
