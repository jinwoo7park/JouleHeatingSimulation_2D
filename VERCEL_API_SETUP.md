# Vercel에서 Fly.io 백엔드 API 호출 설정 가이드

Vercel에 배포된 프론트엔드에서 Fly.io에 배포된 백엔드 API를 호출하는 방법입니다.

## 설정 방법

### 1. 환경 변수 설정

#### Vercel 대시보드에서 설정 (추천)

1. **Vercel 대시보드 접속**
   - https://vercel.com/dashboard
   - 프로젝트 선택

2. **Settings로 이동**
   - 상단 메뉴에서 `Settings` 클릭

3. **Environment Variables 추가**
   - 왼쪽 메뉴에서 `Environment Variables` 클릭
   - 또는 직접 URL: `https://vercel.com/[프로젝트명]/settings/environment-variables`

4. **환경 변수 추가**
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://jouleheatingsimulation-2d.fly.dev`
   - **Environment**: 모든 환경에 적용 (Production, Preview, Development)
   - `Add` 버튼 클릭

5. **재배포**
   - 환경 변수 추가 후 자동으로 재배포되거나
   - 수동으로 `Deployments` → `Redeploy` 클릭

#### 로컬 개발 환경 설정

프로젝트 루트에 `.env.local` 파일 생성:

```bash
# .env.local
VITE_API_BASE_URL=https://jouleheatingsimulation-2d.fly.dev
```

또는 개발 환경에서는 Vite proxy를 사용 (이미 설정됨):

- 개발 서버 실행 시: `localhost:5000`으로 프록시됨
- 빌드/프로덕션: 환경 변수 또는 Fly.io URL 사용

### 2. 코드 변경 사항

`src/App.jsx`에서 API 호출 부분이 이미 수정되었습니다:

```javascript
// API URL 설정
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
                     (import.meta.env.DEV ? '' : 'https://jouleheatingsimulation-2d.fly.dev')
const apiUrl = `${API_BASE_URL}/api/simulate`
```

**동작 방식:**
- 환경 변수 `VITE_API_BASE_URL`이 있으면 그것을 사용
- 개발 환경(`DEV=true`)에서는 빈 문자열 (상대 경로 `/api` → Vite proxy 사용)
- 프로덕션 환경에서는 Fly.io URL 사용

### 3. CORS 설정 확인

Fly.io 백엔드에서 CORS가 이미 설정되어 있는지 확인:

`app.py`에 다음이 포함되어 있습니다:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용
```

따라서 Vercel에서 배포된 프론트엔드에서도 자동으로 작동합니다.

## 테스트

### 로컬에서 테스트

1. **Fly.io 백엔드가 실행 중인지 확인**
   ```bash
   curl https://jouleheatingsimulation-2d.fly.dev/
   ```

2. **프론트엔드 실행**
   ```bash
   pnpm dev
   ```

3. **브라우저에서 테스트**
   - 시뮬레이션 실행
   - 네트워크 탭에서 API 요청 확인
   - `https://jouleheatingsimulation-2d.fly.dev/api/...`로 요청이 가는지 확인

### Vercel 배포 후 테스트

1. **Vercel에 배포**
   ```bash
   git add .
   git commit -m "Add Fly.io API configuration"
   git push
   ```

2. **환경 변수 설정 확인**
   - Vercel 대시보드 → Settings → Environment Variables
   - `VITE_API_BASE_URL`이 설정되어 있는지 확인

3. **배포 후 테스트**
   - 배포된 Vercel URL로 접속
   - 시뮬레이션 실행
   - 브라우저 개발자 도구 → Network 탭에서 API 요청 확인

## 문제 해결

### CORS 오류 발생 시

**증상:**
```
Access to fetch at 'https://jouleheatingsimulation-2d.fly.dev/api/...' from origin 'https://your-app.vercel.app' has been blocked by CORS policy
```

**해결:**
- `app.py`에서 `CORS(app)` 설정이 있는지 확인
- 또는 특정 도메인만 허용:
  ```python
  CORS(app, origins=["https://your-app.vercel.app"])
  ```

### 환경 변수가 적용되지 않을 때

**확인 사항:**
1. Vercel에서 환경 변수 이름이 `VITE_API_BASE_URL`인지 확인 (오타 없이)
2. 재배포가 완료되었는지 확인
3. 브라우저 개발자 도구 → Console에서 환경 변수 확인:
   ```javascript
   console.log(import.meta.env.VITE_API_BASE_URL)
   ```

### API 요청이 실패할 때

1. **Fly.io 백엔드 상태 확인**
   ```bash
   curl https://jouleheatingsimulation-2d.fly.dev/
   ```

2. **네트워크 탭에서 오류 메시지 확인**
   - 404: 경로 확인
   - 500: 서버 로그 확인 (Fly.io → Logs & Errors)
   - 타임아웃: Fly.io 머신이 중지되었을 수 있음 (자동 시작 대기)

## 현재 설정 요약

- **백엔드 URL**: `https://jouleheatingsimulation-2d.fly.dev`
- **API 엔드포인트**:
  - `POST /api/simulate` - 시뮬레이션 시작
  - `GET /api/progress/:session_id` - 진행률 조회
  - `GET /api/result/:session_id` - 결과 다운로드

## 다음 단계

1. ✅ Vercel 환경 변수 설정
2. ✅ 코드 변경 완료 (이미 적용됨)
3. ✅ 재배포
4. ✅ 테스트

