@Library('ml-lib') _

pipeline {
    agent any
    environment {
        PATH = "C:\\Users\\Raja Ehtasham\\mlops-deployment\\venv\\Scripts;${env.PATH}"
        PYTHONUTF8 = "1"
        PYTHONIOENCODING = "utf-8"
    }
    stages {
        stage('Dev Pipeline') {
            when { branch 'dev' }
            steps { script { mlPipeline.dev() } }
        }
        stage('Pre-Prod Pipeline') {
            when { branch 'main' }
            steps { script { mlPipeline.preprod() } }
        }
        stage('Prod Pipeline') {
            when { buildingTag() }
            steps { script { mlPipeline.prod() } }
        }
    }
    post {
        success {
            emailext subject: "SUCCESS: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                     body: "Pipeline passed: ${env.BUILD_URL}",
                     to: "rajaehtasham44@gmail.com"
        }
        failure {
            emailext subject: "FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                     body: "Pipeline failed: ${env.BUILD_URL}",
                     to: "rajaehtasham44@gmail.com"
        }
    }
}
