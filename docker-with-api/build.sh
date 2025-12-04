#!/bin/bash

# ============================================
# Build script for DeepAnalyze Docker Environment with API
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="deepanalyze-env-with-api"
IMAGE_TAG="latest"
CONTAINER_NAME="deepanalyze-full-stack"
DOCKERFILE="Dockerfile"

# Parse command line arguments
USE_ALTERNATIVE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --alternative|-a)
            USE_ALTERNATIVE=true
            DOCKERFILE="Dockerfile.alternative"
            IMAGE_NAME="deepanalyze-env-with-api-alt"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --alternative, -a  Use alternative PyTorch base image (more stable)"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🔨 Building DeepAnalyze Docker Image with API Support${NC}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Error: Dockerfile not found in current directory${NC}"
    echo "Please run this script from the directory containing the Dockerfile"
    exit 1
fi

# Copy API directory if it doesn't exist in current directory
if [ ! -d "./API" ]; then
    if [ -d "../API" ]; then
        echo -e "${YELLOW}📋 Copying API directory to build context...${NC}"
        cp -r ../API .
        echo -e "${GREEN}✅ API directory copied${NC}"
    else
        echo -e "${RED}❌ Error: API directory not found${NC}"
        echo "Please ensure the API directory exists in ../API or ./API"
        exit 1
    fi
else
    echo -e "${GREEN}✅ API directory found in build context${NC}"
fi

echo -e "${YELLOW}📋 Build Configuration:${NC}"
echo "  - Image name: ${IMAGE_NAME}"
echo "  - Tag: ${IMAGE_TAG}"
echo "  - Container name: ${CONTAINER_NAME}"
echo "  - Dockerfile: ${DOCKERFILE}"
if [ "$USE_ALTERNATIVE" = true ]; then
    echo "  - Base image: PyTorch (more stable)"
else
    echo "  - Base image: NVIDIA CUDA (original)"
fi
echo ""

# Stop and remove existing container if it exists
echo -e "${YELLOW}🛑 Stopping existing container (if any)...${NC}"
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker stop "${CONTAINER_NAME}" || true
    docker rm "${CONTAINER_NAME}" || true
    echo -e "${GREEN}✅ Existing container removed${NC}"
else
    echo -e "${GREEN}✅ No existing container found${NC}"
fi

# Build the Docker image
echo -e "${YELLOW}🏗️  Building Docker image...${NC}"
echo "This may take 10-20 minutes depending on your internet connection..."

docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    .

echo -e "${GREEN}✅ Docker image built successfully!${NC}"

# Show the built image
echo ""
echo -e "${BLUE}📋 Built Docker images:${NC}"
docker images | grep "${IMAGE_NAME}" || true

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📖 Next steps:${NC}"
echo "1. Create a workspace directory: mkdir -p ./workspace"
echo "2. Place your DeepAnalyze-8B model in ./models/ directory"
echo "3. Run with docker-compose: docker-compose up -d"
echo ""
echo -e "${BLUE}🔗 Service URLs after starting:${NC}"
echo "  - vLLM API: http://localhost:8000"
echo "  - API Server: http://localhost:8200"
echo "  - File Server: http://localhost:8100"
echo ""
echo -e "${YELLOW}📝 To view logs:${NC}"
echo "  docker-compose logs -f"