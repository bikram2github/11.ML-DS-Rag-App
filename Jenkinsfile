pipeline{
    agent any

    stages {
        stage('checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/bikram2github/11.ML-DS-Rag-App.git'
            }
        }
        stage("Create Virtual Environment") {
            steps {
                bat '''
                if exist venv\\Scripts\\python.exe (
                    echo Virtual environment already exists. Skipping creation.
                ) else (
                    echo Creating virtual environment...
                    "C:\\Users\\bikra\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m venv venv
                )
                '''
            }
        }
        stage('Verify Python Version') {
            steps {
                bat '''
                venv\\Scripts\\python --version
                '''
            }
        }
        stage("Install Dependencies") {
            steps {
                bat '.\\venv\\Scripts\\pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu'
                bat '.\\venv\\Scripts\\pip install -r requirements.txt'
            }
        }
        stage('Run Pytest') {
            steps {
                bat '''
                venv\\Scripts\\pytest -v
                '''
            }
        }
        stage('Stop Existing Containers') {
            steps {
                bat '''
                docker compose down || echo "No containers running"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                bat '''
                docker compose build
                '''
            }
        }

        stage('Start Application') {
            steps {
                bat '''
                docker compose up -d
                '''
            }
        }
    }

    post {
        success {
            echo " Tests passed & app deployed locally"
        }
        failure {
            echo " Tests failed or deployment error"
        }
        always {
            bat 'echo Pipeline finished'
        }
    }
}

