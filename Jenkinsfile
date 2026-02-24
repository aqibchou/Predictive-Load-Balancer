// Predictive Load Balancer — Jenkins CI/CD Pipeline
//
// Two logical paths:
//   PR  (changeRequest) → lint + unit tests + failsafe load gate
//   main (branch 'main') → lint + unit tests + docker build+push + version bump
//
// Jenkins never touches the Kubernetes cluster.
// ArgoCD auto-detects the values.yaml version bump and reconciles.

pipeline {
    agent any

    environment {
        GHCR_REGISTRY = 'ghcr.io'
        LB_IMAGE      = "${GHCR_REGISTRY}/${env.GITHUB_REPOSITORY_OWNER ?: 'YOUR_ORG'}/predictive-lb/load-balancer"
        BACKEND_IMAGE = "${GHCR_REGISTRY}/${env.GITHUB_REPOSITORY_OWNER ?: 'YOUR_ORG'}/predictive-lb/backend"
        IMAGE_TAG     = "${GIT_COMMIT.take(7)}"
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    stages {

        // ── Shared: Checkout ─────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // ── Shared: Lint ─────────────────────────────────────────────────────
        stage('Lint') {
            steps {
                sh '''
                    python3 -m pip install --quiet ruff
                    ruff check load_balancer/ scripts/ jenkins/scripts/ --select E,W,F --ignore E501
                '''
            }
        }

        // ── Shared: Unit Tests ───────────────────────────────────────────────
        stage('Unit Tests') {
            steps {
                sh '''
                    python3 -m pip install --quiet pytest pytest-asyncio grpcio grpcio-tools psutil prometheus-client
                    python3 -m pip install --quiet -r load_balancer/requirements.txt
                    bash telemetry/generate_proto.sh
                    pytest tests/unit/ -v --tb=short --junitxml=results/junit-unit.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'results/junit-unit.xml'
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // PR PATH — Failsafe Load Gate
        // ═══════════════════════════════════════════════════════════════════════

        stage('Build Test Image') {
            when { changeRequest() }
            steps {
                // Build context is project root so COPY telemetry/ works in Dockerfile
                sh '''
                    docker build \
                        -t predictive-lb-test:${BUILD_NUMBER} \
                        -f load_balancer/Dockerfile \
                        .
                '''
            }
        }

        stage('Start Stack') {
            when { changeRequest() }
            steps {
                sh '''
                    docker-compose up -d
                    echo "Stack started — waiting for services to initialize …"
                    sleep 20
                '''
            }
        }

        stage('Failsafe Load Gate') {
            when { changeRequest() }
            steps {
                sh 'bash jenkins/scripts/load_test_gate.sh http://localhost:8000'
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // MAIN BRANCH PATH — Build, Push, Version Bump
        // ═══════════════════════════════════════════════════════════════════════

        stage('Docker Build — Load Balancer') {
            when { branch 'main' }
            steps {
                // Build context is project root so COPY telemetry/ works
                sh '''
                    docker build \
                        -t ${LB_IMAGE}:${IMAGE_TAG} \
                        -t ${LB_IMAGE}:latest \
                        -f load_balancer/Dockerfile \
                        .
                '''
            }
        }

        stage('Docker Build — Backend') {
            when { branch 'main' }
            steps {
                sh '''
                    docker build \
                        -t ${BACKEND_IMAGE}:${IMAGE_TAG} \
                        -t ${BACKEND_IMAGE}:latest \
                        -f backend/Dockerfile \
                        .
                '''
            }
        }

        stage('Push to GHCR') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-credentials',
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                )]) {
                    sh '''
                        echo "${GHCR_TOKEN}" | docker login ${GHCR_REGISTRY} -u "${GHCR_USER}" --password-stdin

                        docker push ${LB_IMAGE}:${IMAGE_TAG}
                        docker push ${LB_IMAGE}:latest

                        docker push ${BACKEND_IMAGE}:${IMAGE_TAG}
                        docker push ${BACKEND_IMAGE}:latest
                    '''
                }
            }
        }

        stage('Version Bump → ArgoCD Trigger') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'github-token',
                    usernameVariable: 'GIT_USER',
                    passwordVariable: 'GIT_TOKEN'
                )]) {
                    sh '''
                        # Update loadBalancer.image.tag in values.yaml
                        sed -i "s|tag: .*|tag: ${IMAGE_TAG}|" k8s/helm/predictive-lb/values.yaml

                        git config user.email "jenkins@ci.local"
                        git config user.name  "Jenkins CI"
                        git add k8s/helm/predictive-lb/values.yaml
                        git commit -m "chore: deploy ${IMAGE_TAG} [skip ci]"

                        git push https://${GIT_USER}:${GIT_TOKEN}@$(
                            git remote get-url origin | sed 's|https://||'
                        ) HEAD:main
                    '''
                }
            }
        }

    } // end stages

    post {
        always {
            // Teardown docker-compose stack (safe even if never started)
            sh 'docker-compose down --remove-orphans || true'
            // Archive Locust report if it was produced
            archiveArtifacts artifacts: 'results/locust_gate*.html,results/locust_gate*.csv',
                             allowEmptyArchive: true
        }
        success {
            script {
                if (env.CHANGE_ID) {
                    githubNotify(
                        status:      'SUCCESS',
                        context:     'jenkins/failsafe',
                        description: "Load gate passed (P95 < 500ms, errors < 1%)"
                    )
                }
            }
        }
        failure {
            script {
                if (env.CHANGE_ID) {
                    githubNotify(
                        status:      'FAILURE',
                        context:     'jenkins/failsafe',
                        description: "Load gate failed — see Jenkins console"
                    )
                }
            }
        }
    }
}
