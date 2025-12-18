FROM node:20-slim

# pnpm 설치
RUN npm install -g pnpm

# 작업 디렉토리 설정
WORKDIR /workspace

# package.json과 pnpm-lock.yaml 복사 (의존성 캐싱 최적화)
COPY package.json pnpm-lock.yaml* ./

# 의존성 설치
RUN pnpm install || pnpm install --no-frozen-lockfile

# 나머지 파일 복사
COPY . .

# 포트 노출
EXPOSE 3000

# 개발 서버 실행
CMD ["pnpm", "dev"]

