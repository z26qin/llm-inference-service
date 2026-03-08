#!/bin/zsh
# =============================================================================
# LLM Inference Service - Minikube Deployment Script
# =============================================================================
# Quick deployment to minikube for local Kubernetes testing
#
# Usage: ./scripts/deploy-minikube.sh [command]
# Commands:
#   setup    - Setup minikube with required addons
#   build    - Build Docker image in minikube
#   deploy   - Deploy all manifests
#   all      - Run all steps (setup + build + deploy)
#   status   - Show deployment status
#   delete   - Delete deployment
#   url      - Get service URL

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
NAMESPACE="llm-inference"
IMAGE_NAME="llm-inference-service"
IMAGE_TAG="latest"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prereqs() {
    log_info "Checking prerequisites..."

    for cmd in minikube kubectl docker; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is required but not installed"
            exit 1
        fi
    done

    log_info "All prerequisites met"
}

# Setup minikube
setup_minikube() {
    log_info "Setting up minikube..."

    # Check if minikube is already running
    if minikube status &> /dev/null; then
        log_warn "Minikube is already running"
    else
        log_info "Starting minikube with 4 CPUs and 8GB memory..."
        minikube start --cpus=4 --memory=8192 --driver=docker
    fi

    log_info "Enabling required addons..."
    minikube addons enable ingress
    minikube addons enable metrics-server

    # Get minikube IP for /etc/hosts
    MINIKUBE_IP=$(minikube ip)
    log_info "Minikube IP: $MINIKUBE_IP"
    log_warn "Add this to /etc/hosts: $MINIKUBE_IP llm.local"

    log_info "Minikube setup complete"
}

# Build image
build_image() {
    log_info "Building Docker image in minikube..."

    # Point to minikube's Docker daemon
    eval $(minikube docker-env)

    cd "$PROJECT_DIR"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} --target production .

    log_info "Image built: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Deploy to Kubernetes
deploy() {
    log_info "Deploying to Kubernetes..."

    cd "$PROJECT_DIR"

    # Apply kustomization
    kubectl apply -k k8s/

    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=llm-inference -n $NAMESPACE --timeout=300s || true

    log_info "Deployment complete"
    show_status
}

# Show status
show_status() {
    echo ""
    log_info "=== Deployment Status ==="
    echo ""

    echo "Pods:"
    kubectl get pods -n $NAMESPACE
    echo ""

    echo "Services:"
    kubectl get svc -n $NAMESPACE
    echo ""

    echo "HPA:"
    kubectl get hpa -n $NAMESPACE 2>/dev/null || echo "No HPA configured"
    echo ""

    echo "Ingress:"
    kubectl get ingress -n $NAMESPACE 2>/dev/null || echo "No Ingress configured"
    echo ""

    # Get URL
    log_info "Access via NodePort:"
    minikube service llm-inference-nodeport -n $NAMESPACE --url 2>/dev/null || echo "Service not ready yet"
}

# Delete deployment
delete_deployment() {
    log_info "Deleting deployment..."

    cd "$PROJECT_DIR"
    kubectl delete -k k8s/ --ignore-not-found

    log_info "Deployment deleted"
}

# Get service URL
get_url() {
    minikube service llm-inference-nodeport -n $NAMESPACE --url
}

# Main
main() {
    check_prereqs

    case "${1:-all}" in
        setup)
            setup_minikube
            ;;
        build)
            build_image
            ;;
        deploy)
            deploy
            ;;
        all)
            setup_minikube
            build_image
            deploy
            ;;
        status)
            show_status
            ;;
        delete)
            delete_deployment
            ;;
        url)
            get_url
            ;;
        *)
            echo "Usage: $0 {setup|build|deploy|all|status|delete|url}"
            exit 1
            ;;
    esac
}

main "$@"
