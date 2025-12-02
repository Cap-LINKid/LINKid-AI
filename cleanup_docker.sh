#!/bin/bash

echo "🧹 Docker & Containerd 정리 시작..."
echo ""

# 1. 현재 상태 확인
echo "📊 현재 Docker 디스크 사용량:"
docker system df
echo ""

echo "📊 Containerd 디렉토리 크기:"
sudo du -sh /var/lib/containerd/ 2>/dev/null || echo "권한 필요"
echo ""

# 2. 실행 중인 컨테이너 확인
echo "🐳 실행 중인 컨테이너:"
docker ps -a
echo ""

# 3. 사용하지 않는 리소스 정리 (안전한 단계)
echo "🧽 1단계: 중지된 컨테이너 제거..."
docker container prune -f

echo "🧽 2단계: 사용하지 않는 이미지 제거..."
docker image prune -f

echo "🧽 3단계: 사용하지 않는 볼륨 제거..."
docker volume prune -f

echo "🧽 4단계: 사용하지 않는 네트워크 제거..."
docker network prune -f

echo ""
echo "📊 정리 후 Docker 디스크 사용량:"
docker system df
echo ""

# 4. 더 강력한 정리 (모든 사용하지 않는 이미지 포함)
read -p "⚠️  모든 사용하지 않는 이미지도 제거하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧽 5단계: 모든 사용하지 않는 이미지 제거 (태그 없는 이미지 포함)..."
    docker image prune -a -f
    echo "완료!"
fi

# 5. containerd 정리 (crictl 사용 가능한 경우)
if command -v crictl &> /dev/null; then
    echo ""
    echo "🧽 6단계: Containerd 이미지 정리..."
    sudo crictl rmi --prune
fi

# 6. 최종 상태 확인
echo ""
echo "📊 최종 Docker 디스크 사용량:"
docker system df
echo ""

echo "📊 최종 Containerd 디렉토리 크기:"
sudo du -sh /var/lib/containerd/ 2>/dev/null || echo "권한 필요"
echo ""

echo "✅ 정리 완료!"


