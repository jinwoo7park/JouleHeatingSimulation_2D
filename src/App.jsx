import React, { useState, useRef } from 'react'
import './App.css'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
  Surface,
  Cell
} from 'recharts'
import * as XLSX from 'xlsx'

const LAYER_NAMES = ['Glass', 'ITO', 'HTL', 'Perovskite', 'ETL', 'Cathode', 'Resin', 'Heat sink']
// 각 레이어의 두께 입력 단위: Glass(mm), Resin(μm), Heat sink(mm), 나머지(nm)
const THICKNESS_UNITS = ['mm', 'nm', 'nm', 'nm', 'nm', 'nm', 'μm', 'mm']
const DEFAULT_VALUES = {
  layer_names: LAYER_NAMES,
  // Resin: UV curable resin (polymer) - k=20, ρ=1100, cp=1800
  // Heat sink: Silicon - k=150, ρ=2330, cp=700
  k_therm_layers: [0.8, 10.0, 0.2, 0.5, 0.2, 200.0, 20.0, 150.0],
  rho_layers: [2500, 7140, 1000, 4100, 1200, 2700, 1100, 2330],
  c_p_layers: [1000, 280, 1500, 250, 1500, 900, 1800, 700],
  // 기본값: Glass=1.1mm, Resin=3μm, Heat sink=1mm, 나머지=nm
  thickness_layers_nm: [1100000, 70, 80, 280, 50, 100, 3000, 1000000], // Resin=3μm, Heat sink=1mm
  layer_enabled: [true, true, true, true, true, true, false, false], // Resin과 Heat sink는 기본적으로 비활성화
  voltage: 2.9,
  current_density: 30.0, // 단위: mA/cm² (기존 300.0 A/m² = 30.0 mA/cm²)
  eqe: 0.2, // External Quantum Efficiency (20%)
  epsilon_top: 0.05,
  epsilon_bottom: 0.85,
  epsilon_side: 0.05, // 측면 방사율
  h_conv: 10.0,
  T_ambient: 25.0, // 섭씨 (°C)
  t_start: 0,
  t_end: 1000.0,
  device_area_mm2: 4.3, // 소자 크기 (mm²)
  r_max_multiplier: 10.0 // r_max = 소자 반지름 × 이 값 (1~100)
}

function App() {
  const [logoError, setLogoError] = useState(false)
  const [formData, setFormData] = useState(DEFAULT_VALUES)
  const [simulationResult, setSimulationResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState({ progress: 0, message: '' })
  const [sessionId, setSessionId] = useState(null)
  const chart1Ref = useRef(null)
  const chart2Ref = useRef(null)

  // 두께 단위 변환 함수들
  const convertToNm = (value, unit) => {
    switch(unit) {
      case 'mm': return value * 1000000  // mm to nm
      case 'μm': return value * 1000      // μm to nm
      case 'nm': return value             // nm
      default: return value
    }
  }
  
  const convertFromNm = (valueNm, unit) => {
    switch(unit) {
      case 'mm': return valueNm / 1000000  // nm to mm
      case 'μm': return valueNm / 1000    // nm to μm
      case 'nm': return valueNm           // nm
      default: return valueNm
    }
  }

  const handleLayerChange = (index, field, value) => {
    const newFormData = { ...formData }
    if (field === 'thickness_layers_nm') {
      // 사용자 입력값을 nm로 변환
      const unit = THICKNESS_UNITS[index]
      const valueInNm = convertToNm(parseFloat(value) || 0, unit)
      newFormData[field][index] = valueInNm
    } else {
      newFormData[field][index] = parseFloat(value) || 0
    }
    setFormData(newFormData)
    // 입력값이 변경되면 이전 시뮬레이션 결과 초기화
    if (simulationResult) {
      setSimulationResult(null)
    }
  }

  const handleGlobalChange = (field, value) => {
    setFormData({ ...formData, [field]: parseFloat(value) || 0 })
    // 입력값이 변경되면 이전 시뮬레이션 결과 초기화
    if (simulationResult) {
      setSimulationResult(null)
    }
  }

  const handleResetToDefault = () => {
    setFormData(DEFAULT_VALUES)
    setSimulationResult(null)
    setError(null)
  }

  // 레이어 활성화/비활성화 핸들러
  const handleLayerEnabledChange = (index, enabled) => {
    const newFormData = { ...formData }
    newFormData.layer_enabled[index] = enabled
    
    // Heat sink (인덱스 7)를 선택하면 Resin (인덱스 6)도 자동 선택
    if (index === 7 && enabled) {
      newFormData.layer_enabled[6] = true
    }
    
    // Resin (인덱스 6)를 해제하면 Heat sink (인덱스 7)도 자동 해제
    if (index === 6 && !enabled) {
      newFormData.layer_enabled[7] = false
    }
    
    setFormData(newFormData)
    // 레이어 활성화 상태가 변경되면 이전 시뮬레이션 결과 초기화
    if (simulationResult) {
      setSimulationResult(null)
    }
  }

  // 섭씨 <-> 켈빈 변환 함수
  const celsiusToKelvin = (celsius) => celsius + 273.15
  const kelvinToCelsius = (kelvin) => kelvin - 273.15

  const handleSimulate = async () => {
    setLoading(true)
    setError(null)
    setSimulationResult(null) // 이전 시뮬레이션 결과 초기화
    setProgress({ progress: 0, message: '시작 중...' })
    console.log('시뮬레이션 시작...')
    
    // 진행률 폴링을 위한 변수 (함수 스코프에서 접근 가능하도록)
    let progressInterval = null
    
    try {
      // 선택된 레이어만 필터링
      const enabledIndices = formData.layer_enabled.map((enabled, idx) => enabled ? idx : -1).filter(idx => idx !== -1)
      const filteredLayerNames = enabledIndices.map(idx => formData.layer_names[idx])
      const filteredK = enabledIndices.map(idx => formData.k_therm_layers[idx])
      const filteredRho = enabledIndices.map(idx => formData.rho_layers[idx])
      const filteredCp = enabledIndices.map(idx => formData.c_p_layers[idx])
      const filteredThickness = enabledIndices.map(idx => formData.thickness_layers_nm[idx])
      
      // 섭씨를 켈빈으로 변환하여 백엔드에 전송
      const dataToSend = {
        ...formData,
        layer_names: filteredLayerNames,
        k_therm_layers: filteredK,
        rho_layers: filteredRho,
        c_p_layers: filteredCp,
        thickness_layers_nm: filteredThickness,
        layer_enabled: formData.layer_enabled,
        T_ambient: celsiusToKelvin(formData.T_ambient),
        device_area_mm2: formData.device_area_mm2,
        r_max_multiplier: formData.r_max_multiplier
      }
      
      // API URL 설정: 환경 변수가 있으면 사용, 없으면 개발 환경에서는 /api, 프로덕션에서는 Fly.io URL
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
                           (import.meta.env.DEV ? '' : 'https://jouleheatingsimulation-2d.fly.dev')
      const apiUrl = `${API_BASE_URL}/api/simulate`
      
      console.log('API 요청 전송 중...', { url: apiUrl, dataSize: JSON.stringify(dataToSend).length })
      
      // 시뮬레이션 요청 전송 (session_id만 받음)
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      })
      
      console.log('API 응답 받음:', { status: response.status, statusText: response.statusText, ok: response.ok })
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('서버 오류 응답:', errorText)
        let errorData
        try {
          errorData = JSON.parse(errorText)
          console.error('파싱된 오류 데이터:', errorData)
        } catch (parseError) {
          console.error('JSON 파싱 오류:', parseError)
          errorData = { error: errorText || `서버 오류: ${response.status} ${response.statusText}` }
        }
        
        // 상세한 오류 정보 표시
        let errorMessage = errorData.error || `서버 오류: ${response.status} ${response.statusText}`
        if (errorData.error_details) {
          console.error('상세 오류 정보:', errorData.error_details)
          if (errorData.error_details.traceback) {
            console.error('전체 Traceback:', errorData.error_details.traceback)
            const tracebackLines = errorData.error_details.traceback.split('\n')
            const relevantLines = tracebackLines.filter(line => 
              line.includes('api/simulate.py') || line.includes('File') || line.includes('line ')
            )
            if (relevantLines.length > 0) {
              errorMessage += `\n\n오류 위치:\n${relevantLines.slice(0, 3).join('\n')}`
            }
          }
          if (errorData.error_details.error_type) {
            errorMessage = `[${errorData.error_details.error_type}] ${errorMessage}`
          }
        }
        console.error('최종 오류 메시지:', errorMessage)
        setError(errorMessage)
        setLoading(false)
        return
      }
      
      let initData
      try {
        const responseText = await response.text()
        if (!responseText || responseText.trim() === '') {
          setError('서버에서 빈 응답을 받았습니다.')
          setLoading(false)
          return
        }
        
        initData = JSON.parse(responseText)
        console.log('초기 응답 파싱 성공, session_id:', initData?.session_id)
      } catch (jsonError) {
        console.error('JSON 파싱 오류:', jsonError)
        setError(`서버 응답 파싱 오류: ${jsonError.message}\n\n브라우저 콘솔을 확인하세요.`)
        setLoading(false)
        return
      }
      
      if (!initData || !initData.success || !initData.session_id) {
        setError('서버에서 session_id를 받지 못했습니다.')
        setLoading(false)
        return
      }
      
      const sessionId = initData.session_id
      setSessionId(sessionId)
      console.log('시뮬레이션 시작됨, session_id:', sessionId)
      
      // 진행률 폴링 시작
      if (progressInterval) clearInterval(progressInterval)
      // API_BASE_URL은 이미 위에서 선언됨
      progressInterval = setInterval(async () => {
        try {
          const progressResponse = await fetch(`${API_BASE_URL}/api/progress/${sessionId}`)
          if (!progressResponse.ok) {
            console.warn('진행률 조회 실패:', progressResponse.status)
            return
          }
          const progressData = await progressResponse.json()
          setProgress({ progress: progressData.progress || 0, message: progressData.message || '' })
          
          // 오류가 있으면 표시
          if (progressData.error) {
            setError(progressData.error)
            if (progressInterval) clearInterval(progressInterval)
            setLoading(false)
            return
          }
          
          // 결과가 있으면 처리
          if (progressData.result && progressData.progress >= 100) {
            console.log('시뮬레이션 완료, 데이터 변환 중...')
            const result = progressData.result
            
            try {
              // 켈빈을 섭씨로 변환하여 저장
              const convertedData = {
                ...result,
                // 2D 데이터 처리
                temperature_2d: result.temperature_2d ? result.temperature_2d.map(row => 
                  Array.isArray(row) ? row.map(kelvin => {
                    const celsius = kelvinToCelsius(kelvin)
                    if (celsius < -200 || celsius > 200) {
                      console.warn('비정상적인 온도 값:', { kelvin, celsius })
                    }
                    return celsius
                  }) : row
                ) : null,
                // temperature_center는 딕셔너리 리스트일 수 있음
                temperature_center: result.temperature_center ? result.temperature_center.map(row => {
                  // 딕셔너리 형태인 경우 (새로운 형식)
                  if (row && typeof row === 'object' && 'temperature' in row) {
                    return {
                      ...row,
                      temperature: Array.isArray(row.temperature) ? row.temperature.map(kelvin => kelvinToCelsius(kelvin)) : row.temperature
                    }
                  }
                  // 배열 형태인 경우 (하위 호환성)
                  if (Array.isArray(row)) {
                    return row.map(kelvin => kelvinToCelsius(kelvin))
                  }
                  return row
                }) : null,
                // 1D 데이터 (하위 호환성)
                temperature_active: result.temperature_active ? result.temperature_active.map(row => 
                  Array.isArray(row) ? row.map(kelvin => kelvinToCelsius(kelvin)) : row
                ) : null,
                temperature_glass: result.temperature_glass ? result.temperature_glass.map(row => 
                  Array.isArray(row) ? row.map(kelvin => kelvinToCelsius(kelvin)) : row
                ) : null,
                perovskite_center_temp: result.perovskite_center_temp ? result.perovskite_center_temp.map(kelvin => {
                  const celsius = kelvinToCelsius(kelvin)
                  if (celsius < -200 || celsius > 200) {
                    console.warn('비정상적인 perovskite 온도:', { kelvin, celsius })
                  }
                  return celsius
                }) : [],
                // r 방향 프로파일 변환 (켈빈 -> 섭씨)
                temp_profile_z_perovskite_r: result.temp_profile_z_perovskite_r ? result.temp_profile_z_perovskite_r.map(kelvin => {
                  const celsius = kelvinToCelsius(kelvin)
                  if (celsius < -200 || celsius > 200) {
                    console.warn('비정상적인 temp_profile_z_perovskite_r 온도:', { kelvin, celsius })
                  }
                  return celsius
                }) : null,
                // z 방향 프로파일 변환 (켈빈 -> 섭씨)
                temp_profile_r0_z: result.temp_profile_r0_z ? result.temp_profile_r0_z.map(kelvin => {
                  const celsius = kelvinToCelsius(kelvin)
                  if (celsius < -200 || celsius > 200) {
                    console.warn('비정상적인 temp_profile_r0_z 온도:', { kelvin, celsius })
                  }
                  return celsius
                }) : null,
                // r=0에서 z, time에 따른 온도 변환 (켈빈 -> 섭씨)
                temp_profile_r0_z_time: result.temp_profile_r0_z_time ? result.temp_profile_r0_z_time.map(zRow => 
                  Array.isArray(zRow) ? zRow.map(kelvin => kelvinToCelsius(kelvin)) : zRow
                ) : null,
                z_profile_nm_sampled: result.z_profile_nm_sampled || null
              }
              
              // 디버깅: 변환된 온도 값 확인
              if (convertedData.temp_profile_z_perovskite_r && convertedData.temp_profile_z_perovskite_r.length > 0) {
                console.log('temp_profile_z_perovskite_r 변환 확인:', {
                  원본_켈빈: result.temp_profile_z_perovskite_r?.slice(0, 3),
                  변환_섭씨: convertedData.temp_profile_z_perovskite_r.slice(0, 3),
                  최대_섭씨: Math.max(...convertedData.temp_profile_z_perovskite_r),
                  최소_섭씨: Math.min(...convertedData.temp_profile_z_perovskite_r)
                })
              }
              
              console.log('변환 완료, 결과 설정 중...')
              setSimulationResult(convertedData)
              setProgress({ progress: 100, message: '완료!' })
              console.log('✅ 시뮬레이션 결과가 화면에 표시됩니다.')
              // 진행률 폴링 중지 및 로딩 상태 해제
              if (progressInterval) clearInterval(progressInterval)
              setLoading(false)
            } catch (conversionError) {
              console.error('데이터 변환 오류:', conversionError)
              setError(`데이터 변환 중 오류가 발생했습니다: ${conversionError.message}`)
              if (progressInterval) clearInterval(progressInterval)
              setLoading(false)
            }
          }
        } catch (err) {
          console.error('진행률 조회 오류:', err)
          // 오류가 발생해도 폴링 계속 (네트워크 일시적 오류일 수 있음)
        }
      }, 200) // 200ms마다 조회
    } catch (err) {
      console.error('API 호출 오류:', err)
      console.error('오류 상세:', err.stack)
      setError(`서버에 연결할 수 없습니다: ${err.message || '네트워크 오류가 발생했습니다.'}\n\n상세: ${err.stack || '알 수 없는 오류'}`)
      setSimulationResult(null)  // 결과 초기화
    } finally {
      // 진행률 폴링은 데이터 변환 완료 후에만 중지
      // (finally 블록에서 즉시 중지하지 않음)
      console.log('API 요청 완료 (finally 블록)')
    }
  }

  // 히트맵 데이터 준비
  const prepareHeatmapData = () => {
    if (!simulationResult) return []
    
    const data = []
    const { time, position_nm, temperature } = simulationResult
    
    for (let i = 0; i < time.length; i++) {
      for (let j = 0; j < position_nm.length; j++) {
        data.push({
          time: time[i],
          position: position_nm[j],
          temperature: temperature[j][i]
        })
      }
    }
    return data
  }

  // Glass 물결선 데이터 (2D에서는 사용하지 않을 수 있음)
  const getGlassWavyProfile = () => {
    if (!simulationResult) return []
    
    const { time, temperature_glass } = simulationResult
    const finalTimeIndex = time.length - 1
    
    if (!temperature_glass || temperature_glass.length === 0) return []
    
    const glassStartTemp = temperature_glass[0][finalTimeIndex]
    const glassEndTemp = temperature_glass[temperature_glass.length - 1][finalTimeIndex]
    const nPoints = 50
    const result = []
    
    for (let i = 0; i <= nPoints; i++) {
      const x = -200 + (200 / nPoints) * i
      const tBase = glassStartTemp + (glassEndTemp - glassStartTemp) * (i / nPoints)
      // 물결 모양 추가
      const amplitude = Math.abs(glassEndTemp - glassStartTemp) * 0.02
      const wave = amplitude * Math.sin((i / nPoints) * 4 * Math.PI)
      result.push({
        position: x,
        temperature: tBase + wave
      })
    }
    
    return result
  }
  
  // 활성층 온도 프로파일 데이터 (r=0에서의 z 방향 프로파일)
  const getActiveProfile = () => {
    if (!simulationResult) return []
    
    const { time, z_profile_nm, temp_profile_r0_z, position_active_nm, temperature_center, temperature_active, glass_ito_boundary_nm, layer_boundaries_nm } = simulationResult
    const finalTimeIndex = time.length - 1
    
    // z_profile_nm과 temp_profile_r0_z가 있으면 우선 사용 (전체 z 좌표)
    if (z_profile_nm && temp_profile_r0_z && z_profile_nm.length === temp_profile_r0_z.length) {
      let profile = z_profile_nm.map((z, idx) => ({
        position: z,
        temperature: temp_profile_r0_z[idx]
      }))
      
      // ITO부터 Cathode까지의 영역만 필터링
      if (glass_ito_boundary_nm !== undefined && layer_boundaries_nm && layer_boundaries_nm.length > 0) {
        const startZ = glass_ito_boundary_nm // ITO 시작점
        const endZ = glass_ito_boundary_nm + layer_boundaries_nm[layer_boundaries_nm.length - 1] // Cathode 끝점
        
        profile = profile.filter(point => point.position >= startZ && point.position <= endZ)
        
        // ITO 시작점을 0으로 재조정 (상대 위치)
        if (profile.length > 0) {
          profile = profile.map(point => ({
            position: point.position - startZ, // ITO 시작점을 0으로
            temperature: point.temperature
          }))
        }
      }
      
      return profile
    }
    
    // 2D 데이터가 있으면 사용, 없으면 1D 데이터 사용
    const tempData = temperature_center || temperature_active
    if (!position_active_nm || !tempData) return []
    
    // temperature_center가 딕셔너리 리스트인 경우 처리
    if (temperature_center && temperature_center.length > 0 && temperature_center[0] && typeof temperature_center[0] === 'object' && 'temperature' in temperature_center[0]) {
      return temperature_center.map(item => ({
        position: item.position_nm || 0,
        temperature: Array.isArray(item.temperature) && item.temperature[finalTimeIndex] !== undefined 
          ? item.temperature[finalTimeIndex] 
          : (item.temperature || null)
      }))
    }
    
    // 배열 형태인 경우 (하위 호환성)
    return position_active_nm.map((pos, idx) => ({
      position: pos,
      temperature: tempData[idx] && Array.isArray(tempData[idx]) ? tempData[idx][finalTimeIndex] : (tempData[idx] || null)
    }))
  }
  
  // 페로브스카이트 중간 지점의 시간에 따른 온도 데이터
  const getPerovskiteCenterProfile = () => {
    if (!simulationResult) return []
    
    const { time, perovskite_center_temp } = simulationResult
    
    return time.map((t, idx) => ({
      time: t,
      temperature: perovskite_center_temp[idx]
    }))
  }
  
  // 레이어 색상 가져오기 (입력창과 동일한 색상)
  const getLayerColor = (layerIndex) => {
    // Glass는 인덱스 0이지만 그래프에서는 제외되므로, 활성층은 인덱스 1부터 시작
    const adjustedIndex = layerIndex + 1  // ITO는 인덱스 1
    return `hsl(${adjustedIndex * 60}, 70%, 80%)`
  }
  
  // 온도 범위 계산 (레이어 라벨 위치 자동 조정용)
  const getTemperatureRange = () => {
    if (!simulationResult) return { min: 0, max: 100 }
    const activeProfile = getActiveProfile()
    const glassProfile = getGlassWavyProfile()
    const allTemps = [...activeProfile.map(p => p.temperature), ...glassProfile.map(p => p.temperature)]
    const min = Math.min(...allTemps)
    const max = Math.max(...allTemps)
    const range = max - min
    return { min: min - range * 0.1, max: max + range * 0.1, range }
  }
  
  // 레이어 영역 데이터 (ReferenceArea용) - ITO부터 Cathode까지만
  const getLayerAreas = () => {
    if (!simulationResult) return []
    
    const { layer_boundaries_nm, glass_ito_boundary_nm } = simulationResult
    
    // ITO부터 Cathode까지의 레이어만 표시
    if (layer_boundaries_nm && layer_boundaries_nm.length > 0 && glass_ito_boundary_nm !== undefined) {
      const areas = []
      
      // 활성층 레이어들 (ITO부터 시작, ITO 시작점을 0으로)
      for (let i = 0; i < layer_boundaries_nm.length - 1; i++) {
        const x1 = layer_boundaries_nm[i] // 이미 ITO 시작점 기준
        const x2 = layer_boundaries_nm[i + 1]
        areas.push({
          x1: x1,
          x2: x2,
          color: getLayerColor(i),
          name: simulationResult.layer_names[i] || `Layer ${i + 1}`,
          centerX: (x1 + x2) / 2
        })
      }
      
      return areas
    }
    
    // 기존 방식 (하위 호환성)
    const areas = []
    if (!layer_boundaries_nm) return areas
    
    // 활성층 레이어들 (ITO부터 시작, 인덱스 1부터)
    for (let i = 0; i < layer_boundaries_nm.length - 1; i++) {
      areas.push({
        x1: layer_boundaries_nm[i],
        x2: layer_boundaries_nm[i + 1],
        color: getLayerColor(i),
        name: simulationResult.layer_names[i] || `Layer ${i + 1}`,
        centerX: (layer_boundaries_nm[i] + layer_boundaries_nm[i + 1]) / 2
      })
    }
    
    return areas
  }
  
  // 레이어 라벨 데이터 (그래프 위에 표시할 텍스트)
  const getLayerLabels = () => {
    if (!simulationResult) return []
    
    const labels = []
    
    // Glass 라벨 (x=-100, 중간 지점)
    labels.push({
      x: -100,
      name: 'Glass (축약)'
    })
    
    // 활성층 레이어 라벨
    const areas = getLayerAreas()
    areas.forEach(area => {
      labels.push({
        x: area.centerX,
        name: area.name
      })
    })
    
    return labels
  }
  
  // 시뮬레이션 기본 정보 계산 (r=0 기준)
  const getSimulationStats = () => {
    if (!simulationResult) return null
    
    const { temp_profile_r0_z, perovskite_center_temp } = simulationResult
    
    // 시작온도 (주변 온도)
    const startTemp = formData.T_ambient || 25.0
    
    // 최종온도 (페로브스카이트 중간 지점의 마지막 온도, r=0)
    const finalTemp = (perovskite_center_temp && perovskite_center_temp.length > 0) 
      ? perovskite_center_temp[perovskite_center_temp.length - 1] 
      : null
    
    // 소자 내부 최대/최소 온도차이 (Vertical: r=0에서 z 방향)
    let maxTempVertical = null, minTempVertical = null, tempDifferenceVertical = null
    if (temp_profile_r0_z && temp_profile_r0_z.length > 0) {
      try {
        maxTempVertical = Math.max(...temp_profile_r0_z)
        minTempVertical = Math.min(...temp_profile_r0_z)
        tempDifferenceVertical = maxTempVertical - minTempVertical
      } catch (e) {
        console.error('Vertical 온도 차이 계산 오류:', e)
      }
    }
    
    // 소자 내부 최대/최소 온도차이 (Lateral: z=perovskite 중점에서 r=0~소자 반지름까지)
    let maxTempLateral = null, minTempLateral = null, tempDifferenceLateral = null
    const { temp_profile_z_perovskite_r, r_mm, device_radius_mm } = simulationResult
    if (temp_profile_z_perovskite_r && r_mm && device_radius_mm && temp_profile_z_perovskite_r.length > 0 && r_mm.length > 0) {
      try {
        // r=0부터 소자 반지름까지의 데이터만 필터링
        const lateralTemps = []
        for (let i = 0; i < r_mm.length && i < temp_profile_z_perovskite_r.length; i++) {
          if (r_mm[i] <= device_radius_mm) {
            lateralTemps.push(temp_profile_z_perovskite_r[i])
          }
        }
        
        if (lateralTemps.length > 0) {
          maxTempLateral = Math.max(...lateralTemps)
          minTempLateral = Math.min(...lateralTemps)
          tempDifferenceLateral = maxTempLateral - minTempLateral
        }
      } catch (e) {
        console.error('Lateral 온도 차이 계산 오류:', e)
      }
    }
    
    // 하단 끝 온도 (Glass 시작점, r=0, z=0)
    let bottomTemp = null
    if (temp_profile_r0_z && temp_profile_r0_z.length > 0) {
      bottomTemp = temp_profile_r0_z[0] // 첫 번째 값이 z=0 (Glass 시작점)
    }
    
    // 상단 끝 온도 (Cathode 끝, r=0, z=최대)
    let topTemp = null
    if (temp_profile_r0_z && temp_profile_r0_z.length > 0) {
      topTemp = temp_profile_r0_z[temp_profile_r0_z.length - 1] // 마지막 값이 z=최대 (Cathode 끝)
    }
    
    return {
      startTemp,
      finalTemp,
      maxTemp: maxTempVertical,
      minTemp: minTempVertical,
      tempDifference: tempDifferenceVertical, // Vertical (수직) 온도 차이
      tempDifferenceLateral, // Lateral (수평) 온도 차이
      maxTempLateral,
      minTempLateral,
      bottomTemp,
      topTemp
    }
  }
  
  // Excel 저장 함수
  const handleSaveExcel = () => {
    if (!simulationResult) {
      alert('시뮬레이션 결과가 없습니다.')
      return
    }
    
    try {
      const { time, temperature_center, perovskite_center_temp, temp_profile_z_perovskite_r, r_mm, z_profile_nm, temp_profile_r0_z, temp_profile_r0_z_time, z_profile_nm_sampled, glass_ito_boundary_nm, layer_boundaries_nm, temperature_2d, position_active_nm } = simulationResult
      const stats = getSimulationStats()
      
      // Sheet1: r=0에서 z, time에 따른 온도 (transpose: 시간이 세로로)
      const sheet1Data = []
      
      console.log('Sheet1 데이터 확인:', {
        has_temp_profile_r0_z_time: !!temp_profile_r0_z_time,
        has_z_profile_nm_sampled: !!z_profile_nm_sampled,
        temp_profile_r0_z_time_length: temp_profile_r0_z_time?.length,
        z_profile_nm_sampled_length: z_profile_nm_sampled?.length,
        time_length: time?.length
      })
      
      if (temp_profile_r0_z_time && z_profile_nm_sampled && time && 
          temp_profile_r0_z_time.length > 0 && z_profile_nm_sampled.length > 0 && time.length > 0) {
        // ITO부터 Cathode까지의 z 위치 필터링
        const startZ = glass_ito_boundary_nm !== undefined ? glass_ito_boundary_nm : 0
        const endZ = (glass_ito_boundary_nm !== undefined && layer_boundaries_nm && layer_boundaries_nm.length > 0) 
          ? glass_ito_boundary_nm + layer_boundaries_nm[layer_boundaries_nm.length - 1] 
          : Infinity
        
        console.log('Sheet1 필터링 범위:', { startZ, endZ, glass_ito_boundary_nm, layer_boundaries_nm })
        
        // ITO~Cathode 범위 내의 z 위치만 필터링
        const filteredIndices = []
        z_profile_nm_sampled.forEach((z, idx) => {
          if (z >= startZ && z <= endZ && idx < temp_profile_r0_z_time.length) {
            filteredIndices.push({ z, idx })
          }
        })
        
        console.log('Sheet1 필터링된 인덱스 수:', filteredIndices.length)
        
        if (filteredIndices.length > 0) {
          // 헤더: 시간 (s), z1, z2, ...
          const headerRow = ['시간 (s)', ...filteredIndices.map(({ z }) => Number(z))]
          sheet1Data.push(headerRow)
          
          // 각 시간별로 z 위치에 따른 온도 데이터 (transpose)
          time.forEach((t, timeIdx) => {
            const row = [Number(t)]
            filteredIndices.forEach(({ idx }) => {
              if (timeIdx < temp_profile_r0_z_time[idx].length) {
                row.push(Number(temp_profile_r0_z_time[idx][timeIdx]))
              } else {
                row.push('')
              }
            })
            sheet1Data.push(row)
          })
        }
      } else if (z_profile_nm && temp_profile_r0_z && z_profile_nm.length === temp_profile_r0_z.length && time) {
        // fallback: z_profile_nm과 temp_profile_r0_z 사용 (최종 시간만, 모든 시간에 대해 동일한 값 사용)
        const startZ = glass_ito_boundary_nm !== undefined ? glass_ito_boundary_nm : 0
        const endZ = (glass_ito_boundary_nm !== undefined && layer_boundaries_nm && layer_boundaries_nm.length > 0) 
          ? glass_ito_boundary_nm + layer_boundaries_nm[layer_boundaries_nm.length - 1] 
          : Infinity
        
        const filteredIndices = []
        z_profile_nm.forEach((z, idx) => {
          if (z >= startZ && z <= endZ) {
            filteredIndices.push({ z, idx })
          }
        })
        
        if (filteredIndices.length > 0) {
          // 헤더: 시간 (s), z1, z2, ...
          const headerRow = ['시간 (s)', ...filteredIndices.map(({ z }) => Number(z))]
          sheet1Data.push(headerRow)
          
          // 모든 시간에 대해 최종 온도 값 사용 (fallback)
          time.forEach((t) => {
            const row = [Number(t), ...filteredIndices.map(({ idx }) => Number(temp_profile_r0_z[idx]))]
            sheet1Data.push(row)
          })
        }
      }
      
      console.log('Sheet1 최종 데이터 행 수:', sheet1Data.length)
      
      // Sheet2: r=0, z=perovskite 중간에서 time에 따른 온도
      const sheet2Data = []
      sheet2Data.push(['시간 (s)', '온도 (°C)'])
      if (perovskite_center_temp && time) {
        console.log('Sheet2 데이터 확인:', {
          perovskite_center_temp_length: perovskite_center_temp.length,
          time_length: time.length
        })
        // 길이가 다를 수 있으므로 최소 길이만큼만 사용
        const minLength = Math.min(perovskite_center_temp.length, time.length)
        for (let idx = 0; idx < minLength; idx++) {
          sheet2Data.push([Number(time[idx]), Number(perovskite_center_temp[idx])])
        }
      }
      
      // Sheet3: z=perovskite 중간에서 r에 따른 온도
      const sheet3Data = []
      sheet3Data.push(['r 위치 (mm)', '온도 (°C)'])
      if (temp_profile_z_perovskite_r && r_mm && temp_profile_z_perovskite_r.length === r_mm.length) {
        r_mm.forEach((r, idx) => {
          sheet3Data.push([Number(r), Number(temp_profile_z_perovskite_r[idx])])
        })
      }
      
      // Sheet4: r-z에 따른 최종온도
      const sheet4Data = []
      if (temperature_2d && r_mm && position_active_nm && temperature_2d.length > 0) {
        // 헤더 행: 첫 번째 셀은 'r (mm) \\ z (μm)', 그 다음 z 좌표들
        const headerRow = ['r (mm) \\ z (μm)']
        position_active_nm.forEach((z_nm) => {
          headerRow.push(Number(z_nm / 1e6)) // nm를 μm로 변환하여 표시
        })
        sheet4Data.push(headerRow)
        
        // 데이터 행: 각 r 좌표를 첫 번째 열에, 그 다음 온도 값들
        r_mm.forEach((r, rIdx) => {
          if (rIdx < temperature_2d.length) {
            const row = [Number(r)] // 첫 번째 열: r 좌표 (mm)
            if (temperature_2d[rIdx] && Array.isArray(temperature_2d[rIdx])) {
              temperature_2d[rIdx].forEach((temp) => {
                row.push(Number(temp))
              })
            }
            sheet4Data.push(row)
          }
        })
      } else {
        // 데이터가 없는 경우 빈 시트
        sheet4Data.push(['r-z 데이터가 없습니다.'])
      }
      
      // Sheet5: 입력 파라미터 및 시뮬레이션 요약
      const sheet5Data = []
      
      // 시뮬레이션 요약
      sheet5Data.push(['시뮬레이션 요약', ''])
      sheet5Data.push(['시작 온도 (°C)', Number(stats.startTemp)])
      sheet5Data.push(['최종 온도 (°C)', stats.finalTemp !== null ? Number(stats.finalTemp) : 'N/A'])
      sheet5Data.push(['하단 끝 온도 (Glass 시작점) (°C)', stats.bottomTemp !== null ? Number(stats.bottomTemp) : 'N/A'])
      sheet5Data.push(['상단 끝 온도 (Cathode 끝) (°C)', stats.topTemp !== null ? Number(stats.topTemp) : 'N/A'])
      sheet5Data.push(['소자 내부 최대 온도 (°C)', stats.maxTemp !== null ? Number(stats.maxTemp) : 'N/A'])
      sheet5Data.push(['소자 내부 최소 온도 (°C)', stats.minTemp !== null ? Number(stats.minTemp) : 'N/A'])
      sheet5Data.push(['소자 내부 온도 차이 (Vertical) (°C)', stats.tempDifference !== null ? Number(stats.tempDifference) : 'N/A'])
      sheet5Data.push(['소자 내부 온도 차이 (Lateral) (°C)', stats.tempDifferenceLateral !== null ? Number(stats.tempDifferenceLateral) : 'N/A'])
      
      // 빈 행 추가
      sheet5Data.push([])
      
      // 입력 파라미터 추가
      sheet5Data.push(['입력 파라미터', ''])
      sheet5Data.push(['레이어 이름', '두께 (단위별)', '열전도도 (W/m·K)', '밀도 (kg/m³)', '비열 (J/kg·K)'])
      
      // 활성화된 레이어만 출력
      LAYER_NAMES.forEach((name, idx) => {
        if (formData.layer_enabled && formData.layer_enabled[idx]) {
          sheet5Data.push([
            name,
            `${convertFromNm(formData.thickness_layers_nm[idx], THICKNESS_UNITS[idx])} ${THICKNESS_UNITS[idx]}`,
            Number(formData.k_therm_layers[idx]),
            Number(formData.rho_layers[idx]),
            Number(formData.c_p_layers[idx])
          ])
        }
      })
      
      // 빈 행 추가
      sheet5Data.push([])
      
      // 전기적 파라미터
      sheet5Data.push(['전기적 파라미터', ''])
      sheet5Data.push(['전압 (V)', Number(formData.voltage)])
      sheet5Data.push(['전류 밀도 (mA/cm²)', Number(formData.current_density)])
      sheet5Data.push(['EQE (External Quantum Efficiency)', Number(formData.eqe)])
      sheet5Data.push(['소자 크기 (mm²)', Number(formData.device_area_mm2)])
      sheet5Data.push(['r_max 배수 (소자 반지름의 배수)', Number(formData.r_max_multiplier)])
      
      // 빈 행 추가
      sheet5Data.push([])
      
      // 열적 파라미터
      sheet5Data.push(['열적 파라미터', ''])
      sheet5Data.push(['상부 방사율 (Cathode)', Number(formData.epsilon_top)])
      sheet5Data.push(['하부 방사율 (Glass)', Number(formData.epsilon_bottom)])
      sheet5Data.push(['측면 방사율 (r=R_max)', Number(formData.epsilon_side)])
      sheet5Data.push(['대류 계수 (W/m²·K)', Number(formData.h_conv)])
      sheet5Data.push(['주변 온도 (°C)', Number(formData.T_ambient)])
      
      // 빈 행 추가
      sheet5Data.push([])
      
      // 시뮬레이션 시간
      sheet5Data.push(['시뮬레이션 시간', ''])
      sheet5Data.push(['시작 시간 (s)', Number(formData.t_start)])
      sheet5Data.push(['종료 시간 (s)', Number(formData.t_end)])
      
      // 워크북 생성
      const wb = XLSX.utils.book_new()
      
      // Sheet1: r=0에서 z, time에 따른 온도
      const ws1 = XLSX.utils.aoa_to_sheet(sheet1Data)
      XLSX.utils.book_append_sheet(wb, ws1, 'r=0, z-time 온도')
      
      // Sheet2: r=0, z=perovskite 중간에서 time에 따른 온도
      const ws2 = XLSX.utils.aoa_to_sheet(sheet2Data)
      XLSX.utils.book_append_sheet(wb, ws2, 'r=0, z=perovskite, time')
      
      // Sheet3: z=perovskite 중간에서 r에 따른 온도
      const ws3 = XLSX.utils.aoa_to_sheet(sheet3Data)
      XLSX.utils.book_append_sheet(wb, ws3, 'z=perovskite, r 온도')
      
      // Sheet4: r-z에 따른 최종온도
      const ws4 = XLSX.utils.aoa_to_sheet(sheet4Data)
      XLSX.utils.book_append_sheet(wb, ws4, 'r-z 최종온도')
      
      // Sheet5: 입력 파라미터 및 시뮬레이션 요약
      const ws5 = XLSX.utils.aoa_to_sheet(sheet5Data)
      XLSX.utils.book_append_sheet(wb, ws5, '입력 파라미터 및 요약')
      
      // 파일 저장
      const fileName = `simulation_result_${new Date().toISOString().split('T')[0]}.xlsx`
      XLSX.writeFile(wb, fileName)
    } catch (error) {
      console.error('Excel 저장 중 오류:', error)
      alert('Excel 저장 중 오류가 발생했습니다: ' + error.message)
    }
  }

  return (
    <div className="app">
      <div className="container">
        <div className="title-section">
          <div className="title-content">
            <h1>Joule Heating Simulation (2D Cylindrical)</h1>
            <p className="subtitle">Heat dissipation in PeLED operation using 2D cylindrical coordinate heat equation</p>
          </div>
          <img
            src="/PNEL_logo.png"
            alt="PNEL Logo"
            className="title-logo"
            onError={() => setLogoError(true)}
            style={{ display: logoError ? 'none' : 'block' }}
          />
        </div>

        <div className="simulation-container">
          {/* 소자 구조 입력 섹션 */}
          <div className="input-section">
            <h2>소자 구조 및 물성 입력</h2>
            
            {/* 레이어별 입력 */}
            <div className="layers-container">
              <div className="section-header">
                <h3>레이어 물성</h3>
                <button 
                  className="reset-button" 
                  onClick={handleResetToDefault}
                  title="모든 값을 기본값으로 되돌립니다"
                >
                  기본값으로 되돌리기
                </button>
              </div>
              <div className="layers-grid">
                {LAYER_NAMES.map((name, index) => {
                  const isResin = index === 6
                  const isHeatSink = index === 7
                  const showCheckbox = isResin || isHeatSink
                  
                  return (
                    <div key={index} className="layer-card" style={{ 
                      opacity: showCheckbox && !formData.layer_enabled[index] ? 0.5 : 1 
                    }}>
                      <div className="layer-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%' }}>
                          <h4>{name}</h4>
                          {showCheckbox && (
                            <label style={{ display: 'flex', alignItems: 'center', marginLeft: isResin ? '70px' : '20px', cursor: 'pointer' }}>
                              <input
                                type="checkbox"
                                checked={formData.layer_enabled[index]}
                                onChange={(e) => handleLayerEnabledChange(index, e.target.checked)}
                              />
                            </label>
                          )}
                        </div>
                        <div className="layer-visual" style={{ 
                          height: `${Math.max(30, Math.log10(formData.thickness_layers_nm[index] + 1) * 10)}px`,
                          backgroundColor: `hsl(${index * 60}, 70%, 80%)`
                        }}></div>
                      </div>
                      <div className="layer-inputs" style={{ 
                        pointerEvents: showCheckbox && !formData.layer_enabled[index] ? 'none' : 'auto' 
                      }}>
                        <div className="input-field">
                          <label>두께 ({THICKNESS_UNITS[index]})</label>
                          <input
                            type="number"
                            value={convertFromNm(formData.thickness_layers_nm[index], THICKNESS_UNITS[index])}
                            onChange={(e) => handleLayerChange(index, 'thickness_layers_nm', e.target.value)}
                            step={THICKNESS_UNITS[index] === 'nm' ? '0.1' : THICKNESS_UNITS[index] === 'μm' ? '0.001' : '0.0001'}
                            disabled={showCheckbox && !formData.layer_enabled[index]}
                          />
                        </div>
                        <div className="input-field">
                          <label>열전도도 (W/m·K)</label>
                          <input
                            type="number"
                            value={formData.k_therm_layers[index]}
                            onChange={(e) => handleLayerChange(index, 'k_therm_layers', e.target.value)}
                            step="0.1"
                            disabled={showCheckbox && !formData.layer_enabled[index]}
                          />
                        </div>
                        <div className="input-field">
                          <label>밀도 (kg/m³)</label>
                          <input
                            type="number"
                            value={formData.rho_layers[index]}
                            onChange={(e) => handleLayerChange(index, 'rho_layers', e.target.value)}
                            step="1"
                            disabled={showCheckbox && !formData.layer_enabled[index]}
                          />
                        </div>
                        <div className="input-field">
                          <label>비열 (J/kg·K)</label>
                          <input
                            type="number"
                            value={formData.c_p_layers[index]}
                            onChange={(e) => handleLayerChange(index, 'c_p_layers', e.target.value)}
                            step="1"
                            disabled={showCheckbox && !formData.layer_enabled[index]}
                          />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 전기적 파라미터 */}
            <div className="parameters-section">
              <h3>전기적 파라미터</h3>
              <div className="parameters-grid">
                <div className="input-field">
                  <label>전압 (V)</label>
                  <input
                    type="number"
                    value={formData.voltage}
                    onChange={(e) => handleGlobalChange('voltage', e.target.value)}
                    step="0.1"
                  />
                </div>
                <div className="input-field">
                  <label>전류 밀도 (mA/cm²)</label>
                  <input
                    type="number"
                    value={formData.current_density}
                    onChange={(e) => handleGlobalChange('current_density', e.target.value)}
                    step="1"
                  />
                </div>
                <div className="input-field">
                  <label>EQE (External Quantum Efficiency)</label>
                  <input
                    type="number"
                    value={formData.eqe}
                    onChange={(e) => handleGlobalChange('eqe', e.target.value)}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>
                <div className="input-field">
                  <label>소자 크기 (mm²)</label>
                  <input
                    type="number"
                    value={formData.device_area_mm2}
                    onChange={(e) => handleGlobalChange('device_area_mm2', e.target.value)}
                    step="0.1"
                    min="0.01"
                  />
                  <small style={{ color: '#666', fontSize: '0.85em' }}>
                    계산된 반지름: {Math.sqrt(formData.device_area_mm2 / Math.PI).toFixed(3)} mm
                  </small>
                </div>
                <div className="input-field">
                  <label>r_max 배수 (소자 반지름의 배수)</label>
                  <input
                    type="number"
                    value={formData.r_max_multiplier}
                    onChange={(e) => handleGlobalChange('r_max_multiplier', e.target.value)}
                    step="1"
                    min="1"
                    max="100"
                  />
                  <small style={{ color: '#666', fontSize: '0.85em' }}>
                    r_max = 소자 반지름 × {formData.r_max_multiplier} = {Math.sqrt(formData.device_area_mm2 / Math.PI) * formData.r_max_multiplier} mm
                  </small>
                </div>
              </div>
            </div>

            {/* 열적 파라미터 */}
            <div className="parameters-section">
              <h3>열적 파라미터</h3>
              <div className="parameters-grid">
                <div className="input-field">
                  <label>상부 방사율 (Cathode/Heat sink)</label>
                  <input
                    type="number"
                    value={formData.epsilon_top}
                    onChange={(e) => handleGlobalChange('epsilon_top', e.target.value)}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>
                <div className="input-field">
                  <label>하부 방사율 (Glass)</label>
                  <input
                    type="number"
                    value={formData.epsilon_bottom}
                    onChange={(e) => handleGlobalChange('epsilon_bottom', e.target.value)}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>
                <div className="input-field">
                  <label>측면 방사율 (r=R_max)</label>
                  <input
                    type="number"
                    value={formData.epsilon_side}
                    onChange={(e) => handleGlobalChange('epsilon_side', e.target.value)}
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>
                <div className="input-field">
                  <label>대류 계수 (W/m²·K)</label>
                  <input
                    type="number"
                    value={formData.h_conv}
                    onChange={(e) => handleGlobalChange('h_conv', e.target.value)}
                    step="0.1"
                  />
                </div>
                <div className="input-field">
                  <label>주변 온도 (°C)</label>
                  <input
                    type="number"
                    value={formData.T_ambient}
                    onChange={(e) => handleGlobalChange('T_ambient', e.target.value)}
                    step="1"
                  />
                </div>
              </div>
            </div>

            {/* 시뮬레이션 시간 설정 */}
            <div className="parameters-section">
              <h3>시뮬레이션 시간</h3>
              <div className="parameters-grid">
                <div className="input-field">
                  <label>시작 시간 (s)</label>
                  <input
                    type="number"
                    value={formData.t_start}
                    onChange={(e) => handleGlobalChange('t_start', e.target.value)}
                    step="0.1"
                  />
                </div>
                <div className="input-field">
                  <label>종료 시간 (s)</label>
                  <input
                    type="number"
                    value={formData.t_end}
                    onChange={(e) => handleGlobalChange('t_end', e.target.value)}
                    step="10"
                  />
                </div>
              </div>
            </div>

            <button 
              className="simulate-button" 
              onClick={handleSimulate}
              disabled={loading}
            >
              {loading ? '시뮬레이션 실행 중...' : '시뮬레이션 실행'}
            </button>
            
            {(loading || (progress.progress > 0 && progress.progress < 100)) && (
              <div style={{ 
                marginTop: '15px', 
                padding: '15px', 
                backgroundColor: '#f0f9ff', 
                borderRadius: '8px',
                border: '1px solid #bae6fd'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '10px',
                  marginBottom: '8px'
                }}>
                  <div style={{ 
                    width: '100%', 
                    height: '20px', 
                    backgroundColor: '#e0e7ff', 
                    borderRadius: '10px',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      width: `${Math.min(progress.progress, 100)}%`, 
                      height: '100%', 
                      backgroundColor: '#3b82f6',
                      transition: 'width 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}>
                      {progress.progress > 5 && `${Math.round(progress.progress)}%`}
                    </div>
                  </div>
                </div>
                <div style={{ 
                  fontSize: '0.9em', 
                  color: '#64748b',
                  textAlign: 'center'
                }}>
                  {progress.message || '처리 중...'}
                </div>
                <div style={{ 
                  fontSize: '0.8em', 
                  color: '#94a3b8',
                  textAlign: 'center',
                  marginTop: '8px'
                }}>
                  💡 결과는 브라우저 콘솔(F12)에서도 확인할 수 있습니다
                </div>
              </div>
            )}

            {error && (
              <div className="error-message" style={{ whiteSpace: 'pre-wrap' }}>
                {error}
              </div>
            )}
          </div>

          {/* 결과 시각화 섹션 */}
          {simulationResult && (
            <div className="results-section">
              <h2>시뮬레이션 결과</h2>
              <div style={{ 
                marginBottom: '15px', 
                padding: '10px', 
                backgroundColor: '#f0fdf4', 
                borderRadius: '6px',
                border: '1px solid #86efac',
                fontSize: '0.9em',
                color: '#166534'
              }}>
                ✅ 시뮬레이션이 완료되었습니다. 결과가 아래에 표시됩니다.
              </div>
              
              {/* 2D 온도 분포 (x-y 평면, z=perovskite 중점) */}
              {simulationResult.temp_profile_z_perovskite_r && simulationResult.r_mm && simulationResult.device_radius_mm && simulationResult.time && simulationResult.time.length > 0 && (
                <div className="chart-container" style={{ marginBottom: '30px' }}>
                  <h3>2D 온도 분포 (x-y 평면, z = Perovskite 중점, t = {simulationResult.time[simulationResult.time.length - 1]?.toFixed(1) || 'N/A'} s)</h3>
                  <div style={{ 
                    padding: '20px', 
                    backgroundColor: '#f8f9fa', 
                    borderRadius: '8px',
                    border: '1px solid #dee2e6'
                  }}>
                    <div style={{ marginBottom: '10px', fontSize: '0.9em', color: '#666' }}>
                      소자 반지름: {simulationResult.device_radius_mm?.toFixed(3) || 'N/A'} mm | 
                      Perovskite 중점 z 위치: {simulationResult.perovskite_mid_z_nm ? (simulationResult.perovskite_mid_z_nm / 1e6).toFixed(2) + ' μm' : 'N/A'}
                    </div>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                      gap: '20px',
                      marginBottom: '20px'
                    }}>
                      {/* 온도 범위 표시 */}
                      <div>
                        <h4 style={{ fontSize: '1em', marginBottom: '10px' }}>온도 범위</h4>
                        <div style={{ fontSize: '0.9em' }}>
                          최대: {Math.max(...simulationResult.temp_profile_z_perovskite_r).toFixed(2)} °C<br/>
                          최소: {Math.min(...simulationResult.temp_profile_z_perovskite_r).toFixed(2)} °C
                        </div>
                      </div>
                    </div>
                    {/* x-y 평면 Contour Plot 시각화 및 r-T 그래프 */}
                    <div style={{ 
                      display: 'flex',
                      gap: '10px',
                      alignItems: 'flex-start'
                    }}>
                      <div style={{ 
                        width: 'calc(50% - 30px)', 
                        height: '500px', 
                        position: 'relative',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        backgroundColor: '#fff'
                      }}>
                        <canvas 
                          id="contour-canvas" 
                          style={{ 
                            width: '100%', 
                            height: '100%',
                            display: 'block'
                          }}
                          ref={(canvas) => {
                            if (!canvas || !simulationResult.temp_profile_z_perovskite_r || !simulationResult.r_mm || !simulationResult.device_radius_mm) return
                            
                            const ctx = canvas.getContext('2d')
                            // 고해상도 디스플레이 지원
                            const dpr = window.devicePixelRatio || 1
                            const width = canvas.offsetWidth
                            const height = canvas.offsetHeight
                            canvas.width = width * dpr
                            canvas.height = height * dpr
                            ctx.scale(dpr, dpr)
                            canvas.style.width = width + 'px'
                            canvas.style.height = height + 'px'
                            
                            const tempProfile = simulationResult.temp_profile_z_perovskite_r
                            const r_mm = simulationResult.r_mm
                            const deviceRadius = simulationResult.device_radius_mm
                            
                            if (!tempProfile || tempProfile.length === 0 || !r_mm || r_mm.length === 0) return
                            
                            // 온도 범위 계산
                            const minTemp = Math.min(...tempProfile)
                            const maxTemp = Math.max(...tempProfile)
                            const tempRange = maxTemp - minTemp
                            
                            // 색상 맵핑 함수
                            const getColor = (temp) => {
                              const ratio = Math.max(0, Math.min(1, (temp - minTemp) / tempRange))
                              let hue = 240 - (ratio * 240)
                              if (hue < 0) hue += 360
                              const saturation = 70 + (ratio * 30)
                              const lightness = 50 + (ratio * 20)
                              return `hsl(${hue}, ${saturation}%, ${lightness}%)`
                            }
                            
                            // 중심점과 스케일 설정
                            const centerX = width / 2
                            const centerY = height / 2
                            const maxR = Math.max(...r_mm)
                            const scale = Math.min(width, height) / 2 / maxR * 0.9 // 90% 사용하여 여백 확보
                            
                            // 그리드로 온도 분포 그리기 (x-y 평면)
                            const pixelSize = 2 // 픽셀 크기 (성능을 위해)
                            
                            // r 값에 따른 온도 보간 함수
                            const getTempAtR = (r) => {
                              if (r <= 0) return tempProfile[0]
                              if (r >= maxR) return tempProfile[tempProfile.length - 1]
                              
                              // r_mm 배열에서 보간
                              for (let i = 0; i < r_mm.length - 1; i++) {
                                if (r >= r_mm[i] && r <= r_mm[i + 1]) {
                                  const ratio = (r - r_mm[i]) / (r_mm[i + 1] - r_mm[i])
                                  return tempProfile[i] + ratio * (tempProfile[i + 1] - tempProfile[i])
                                }
                              }
                              return tempProfile[0]
                            }
                            
                            // 히트맵 그리기 (캔버스 전체를 사용하되 중심점 기준)
                            for (let px = 0; px < width; px += pixelSize) {
                              for (let py = 0; py < height; py += pixelSize) {
                                // 픽셀 좌표를 물리 좌표로 변환 (중심점 기준)
                                const dx = (px - centerX) / scale
                                const dy = (py - centerY) / scale
                                const r = Math.sqrt(dx * dx + dy * dy)
                                
                                // maxR 범위 내에서만 그리기
                                if (r <= maxR) {
                                  const temp = getTempAtR(r)
                                  ctx.fillStyle = getColor(temp)
                                  ctx.fillRect(px, py, pixelSize, pixelSize)
                                }
                              }
                            }
                            
                            // 소자 반지름 원 표시
                            ctx.strokeStyle = '#000'
                            ctx.lineWidth = 3
                            ctx.setLineDash([5, 5])
                            ctx.beginPath()
                            ctx.arc(centerX, centerY, deviceRadius * scale, 0, 2 * Math.PI)
                            ctx.stroke()
                            ctx.setLineDash([])
                            
                            // 소자 반지름 라벨
                            ctx.fillStyle = '#000'
                            ctx.font = 'bold 12px Arial'
                            ctx.textAlign = 'center'
                            ctx.fillText(`소자 반지름: ${deviceRadius.toFixed(3)} mm`, centerX, centerY + deviceRadius * scale + 20)
                            
                            // 축 그리기
                            ctx.strokeStyle = '#333'
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            ctx.moveTo(centerX - maxR * scale, centerY)
                            ctx.lineTo(centerX + maxR * scale, centerY)
                            ctx.moveTo(centerX, centerY - maxR * scale)
                            ctx.lineTo(centerX, centerY + maxR * scale)
                            ctx.stroke()
                            
                            // 축 라벨
                            ctx.fillStyle = '#333'
                            ctx.font = '12px Arial'
                            ctx.textAlign = 'center'
                            ctx.fillText('x (mm)', centerX, height - 10)
                            ctx.save()
                            ctx.translate(15, centerY)
                            ctx.rotate(-Math.PI / 2)
                            ctx.fillText('y (mm)', 0, 0)
                            ctx.restore()
                            
                            // 눈금 표시 (양수와 음수 모두)
                            ctx.font = '10px Arial'
                            const numTicks = 5
                            for (let i = 0; i <= numTicks; i++) {
                              const rValue = (i / numTicks) * maxR
                              const r = rValue * scale
                              
                              // x축 양수 눈금
                              ctx.beginPath()
                              ctx.moveTo(centerX + r, centerY - 5)
                              ctx.lineTo(centerX + r, centerY + 5)
                              ctx.stroke()
                              ctx.textAlign = 'center'
                              ctx.fillText(rValue.toFixed(2), centerX + r, centerY + 18)
                              
                              // x축 음수 눈금 (0 제외)
                              if (i > 0) {
                                ctx.beginPath()
                                ctx.moveTo(centerX - r, centerY - 5)
                                ctx.lineTo(centerX - r, centerY + 5)
                                ctx.stroke()
                                ctx.fillText((-rValue).toFixed(2), centerX - r, centerY + 18)
                              }
                              
                              // y축 양수 눈금
                              ctx.beginPath()
                              ctx.moveTo(centerX - 5, centerY - r)
                              ctx.lineTo(centerX + 5, centerY - r)
                              ctx.stroke()
                              ctx.textAlign = 'right'
                              ctx.fillText(rValue.toFixed(2), centerX - 8, centerY - r + 4)
                              
                              // y축 음수 눈금 (0 제외)
                              if (i > 0) {
                                ctx.beginPath()
                                ctx.moveTo(centerX - 5, centerY + r)
                                ctx.lineTo(centerX + 5, centerY + r)
                                ctx.stroke()
                                ctx.fillText((-rValue).toFixed(2), centerX - 8, centerY + r + 4)
                              }
                              
                              ctx.textAlign = 'center'
                            }
                          }}
                        />
                      </div>
                      {/* 컬러바 (색상 범례) */}
                      <div style={{
                        width: '50px',
                        height: '500px',
                        position: 'relative',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        backgroundColor: '#fff'
                      }}>
                        <canvas
                          id="colorbar-canvas"
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'block'
                          }}
                          ref={(canvas) => {
                            if (!canvas || !simulationResult.temp_profile_z_perovskite_r) return
                            
                            const ctx = canvas.getContext('2d')
                            // 고해상도 디스플레이 지원
                            const dpr = window.devicePixelRatio || 1
                            const width = canvas.offsetWidth
                            const height = canvas.offsetHeight
                            canvas.width = width * dpr
                            canvas.height = height * dpr
                            ctx.scale(dpr, dpr)
                            canvas.style.width = width + 'px'
                            canvas.style.height = height + 'px'
                            
                            const tempProfile = simulationResult.temp_profile_z_perovskite_r
                            const minTemp = Math.min(...tempProfile)
                            const maxTemp = Math.max(...tempProfile)
                            const tempRange = maxTemp - minTemp
                            
                            // 색상 맵핑 함수
                            const getColor = (temp) => {
                              const ratio = Math.max(0, Math.min(1, (temp - minTemp) / tempRange))
                              let hue = 240 - (ratio * 240)
                              if (hue < 0) hue += 360
                              const saturation = 70 + (ratio * 30)
                              const lightness = 50 + (ratio * 20)
                              return `hsl(${hue}, ${saturation}%, ${lightness}%)`
                            }
                            
                            // 그라데이션 그리기
                            const numSteps = 100
                            for (let i = 0; i < numSteps; i++) {
                              const ratio = i / numSteps
                              const temp = minTemp + ratio * tempRange
                              ctx.fillStyle = getColor(temp)
                              const y = height - (i / numSteps) * height
                              const stepHeight = height / numSteps
                              ctx.fillRect(0, y, width, stepHeight)
                            }
                            
                            // 라벨 추가
                            ctx.fillStyle = '#333'
                            ctx.font = '10px Arial'
                            ctx.textAlign = 'center'
                            ctx.save()
                            ctx.translate(width / 2, 10)
                            ctx.fillText(maxTemp.toFixed(1) + '°C', 0, 0)
                            ctx.translate(0, height - 20)
                            ctx.fillText(minTemp.toFixed(1) + '°C', 0, 0)
                            ctx.restore()
                          }}
                        />
                      </div>
                      {/* r에 따른 T 그래프 */}
                      <div style={{ 
                        width: 'calc(50% - 30px)', 
                        height: '500px', 
                        position: 'relative',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        backgroundColor: '#fff',
                        padding: '10px',
                        display: 'flex',
                        flexDirection: 'column'
                      }}>
                        <h4 style={{ marginTop: 0, marginBottom: '10px', fontSize: '1em', textAlign: 'center', flexShrink: 0 }}>
                          r에 따른 온도 분포
                        </h4>
                        <div style={{ flex: 1, minHeight: 0 }}>
                          {simulationResult.r_mm && simulationResult.temp_profile_z_perovskite_r && 
                           simulationResult.r_mm.length > 0 && simulationResult.temp_profile_z_perovskite_r.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart
                                data={simulationResult.r_mm.map((r, idx) => {
                                  if (idx < simulationResult.temp_profile_z_perovskite_r.length) {
                                    return {
                                      r: r,
                                      temperature: simulationResult.temp_profile_z_perovskite_r[idx]
                                    }
                                  }
                                  return null
                                }).filter(item => item !== null)}
                              >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="r" 
                              type="number"
                              label={{ value: 'r (mm)', position: 'insideBottom', offset: -5 }}
                              domain={['dataMin', 'dataMax']}
                            />
                            <YAxis 
                              label={{ value: '온도 (°C)', angle: -90, position: 'insideLeft' }}
                              domain={['auto', 'auto']}
                            />
                            <Tooltip 
                              formatter={(value, name) => [value.toFixed(2) + ' °C', '온도']}
                              labelFormatter={(label) => `r = ${label.toFixed(3)} mm`}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="temperature" 
                              stroke="#dc2626" 
                              strokeWidth={2}
                              dot={false}
                            />
                            {/* 소자 반지름 수직선 표시 */}
                            {simulationResult.device_radius_mm && (
                              <ReferenceLine 
                                x={simulationResult.device_radius_mm} 
                                stroke="#000" 
                                strokeDasharray="5 5"
                                strokeWidth={2}
                                label={{ value: '소자 반지름', position: 'top' }}
                              />
                            )}
                              </LineChart>
                            </ResponsiveContainer>
                          ) : (
                            <div style={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              justifyContent: 'center', 
                              height: '100%',
                              color: '#666',
                              fontSize: '0.9em'
                            }}>
                              데이터를 불러오는 중...
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#666' }}>
                      * 왼쪽: Contour plot에서 파란색은 낮은 온도, 빨간색은 높은 온도를 나타냅니다. 점선 원은 소자 반지름을 나타냅니다.<br/>
                      * 가운데: r에 따른 온도 분포 그래프. 점선은 소자 반지름 위치를 나타냅니다.
                    </div>
                  </div>
                </div>
              )}

              {/* r-z 평면 2D 온도 분포 (최종 시간) */}
              {simulationResult.temperature_2d && simulationResult.r_mm && simulationResult.position_active_nm && simulationResult.time && simulationResult.time.length > 0 && (
                <div className="chart-container" style={{ marginBottom: '30px' }}>
                  <h3>r-z 평면 온도 분포 (t = {simulationResult.time[simulationResult.time.length - 1]?.toFixed(1) || 'N/A'} s)</h3>
                  <div style={{ 
                    padding: '20px', 
                    backgroundColor: '#f8f9fa', 
                    borderRadius: '8px',
                    border: '1px solid #dee2e6'
                  }}>
                    <div style={{ marginBottom: '10px', fontSize: '0.9em', color: '#666' }}>
                      r 방향: 0 ~ {Math.max(...simulationResult.r_mm).toFixed(3)} mm | 
                      z 방향: 0 ~ {Math.max(...simulationResult.position_active_nm).toFixed(2)} nm (활성층 기준)
                    </div>
                    <div style={{ 
                      display: 'flex',
                      gap: '10px',
                      alignItems: 'flex-start'
                    }}>
                      {/* r-z 히트맵 */}
                      <div style={{ 
                        width: 'calc(70% - 30px)', 
                        height: '500px', 
                        position: 'relative',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        backgroundColor: '#fff'
                      }}>
                        <canvas 
                          id="rz-heatmap-canvas" 
                          style={{ 
                            width: '100%', 
                            height: '100%',
                            display: 'block'
                          }}
                          ref={(canvas) => {
                            if (!canvas || !simulationResult.temperature_2d || !simulationResult.r_mm || !simulationResult.position_active_nm) return
                            
                            const ctx = canvas.getContext('2d')
                            // 고해상도 디스플레이 지원
                            const dpr = window.devicePixelRatio || 1
                            const width = canvas.offsetWidth
                            const height = canvas.offsetHeight
                            canvas.width = width * dpr
                            canvas.height = height * dpr
                            ctx.scale(dpr, dpr)
                            canvas.style.width = width + 'px'
                            canvas.style.height = height + 'px'
                            
                            const temp2d = simulationResult.temperature_2d
                            const r_mm = simulationResult.r_mm
                            const z_nm = simulationResult.position_active_nm
                            
                            if (!temp2d || temp2d.length === 0 || !r_mm || r_mm.length === 0 || !z_nm || z_nm.length === 0) return
                            
                            // 온도 범위 계산
                            const allTemps = temp2d.flat()
                            const minTemp = Math.min(...allTemps)
                            const maxTemp = Math.max(...allTemps)
                            const tempRange = maxTemp - minTemp
                            
                            // 자연스러운 색상 맵핑 함수 (파란색 -> 청록색 -> 노란색 -> 빨간색)
                            const getColor = (temp) => {
                              const ratio = Math.max(0, Math.min(1, (temp - minTemp) / tempRange))
                              
                              // 더 부드러운 색상 전환을 위한 커브 적용
                              const smoothRatio = ratio * ratio * (3 - 2 * ratio) // smoothstep 함수
                              
                              // 색상 단계별 정의 (더 자연스러운 그라데이션)
                              let r, g, b
                              
                              if (smoothRatio < 0.25) {
                                // 파란색 -> 청록색 (0 ~ 0.25)
                                const t = smoothRatio / 0.25
                                r = 0
                                g = Math.floor(100 + t * 155)
                                b = Math.floor(200 + t * 55)
                              } else if (smoothRatio < 0.5) {
                                // 청록색 -> 녹색 -> 노란색 (0.25 ~ 0.5)
                                const t = (smoothRatio - 0.25) / 0.25
                                r = Math.floor(0 + t * 255)
                                g = Math.floor(255)
                                b = Math.floor(255 - t * 255)
                              } else if (smoothRatio < 0.75) {
                                // 노란색 -> 주황색 (0.5 ~ 0.75)
                                const t = (smoothRatio - 0.5) / 0.25
                                r = Math.floor(255)
                                g = Math.floor(255 - t * 100)
                                b = 0
                              } else {
                                // 주황색 -> 빨간색 (0.75 ~ 1.0)
                                const t = (smoothRatio - 0.75) / 0.25
                                r = Math.floor(255)
                                g = Math.floor(155 - t * 155)
                                b = 0
                              }
                              
                              return `rgb(${r}, ${g}, ${b})`
                            }
                            
                            // 좌표계 설정
                            const padding = { top: 40, right: 40, bottom: 60, left: 60 }
                            const plotWidth = width - padding.left - padding.right
                            const plotHeight = height - padding.top - padding.bottom
                            
                            const maxR = Math.max(...r_mm)
                            const maxZ = Math.max(...z_nm)
                            const minR = Math.min(...r_mm)
                            const minZ = Math.min(...z_nm)
                            
                            const scaleX = plotWidth / (maxR - minR)
                            const scaleY = plotHeight / (maxZ - minZ)
                            
                            // 양선형 보간 함수
                            const bilinearInterpolate = (r, z) => {
                              // r, z 좌표에 해당하는 인덱스 찾기
                              let i1 = 0, i2 = r_mm.length - 1
                              let j1 = 0, j2 = z_nm.length - 1
                              
                              // r 방향 인덱스 찾기
                              for (let i = 0; i < r_mm.length - 1; i++) {
                                if (r >= r_mm[i] && r <= r_mm[i + 1]) {
                                  i1 = i
                                  i2 = i + 1
                                  break
                                }
                              }
                              
                              // z 방향 인덱스 찾기
                              for (let j = 0; j < z_nm.length - 1; j++) {
                                if (z >= z_nm[j] && z <= z_nm[j + 1]) {
                                  j1 = j
                                  j2 = j + 1
                                  break
                                }
                              }
                              
                              // 경계 처리
                              if (r < r_mm[0]) { i1 = 0; i2 = 0 }
                              if (r > r_mm[r_mm.length - 1]) { i1 = r_mm.length - 1; i2 = r_mm.length - 1 }
                              if (z < z_nm[0]) { j1 = 0; j2 = 0 }
                              if (z > z_nm[z_nm.length - 1]) { j1 = z_nm.length - 1; j2 = z_nm.length - 1 }
                              
                              // 네 모서리 온도 값 가져오기
                              const getTemp = (i, j) => {
                                if (i >= 0 && i < temp2d.length && j >= 0 && j < temp2d[i].length) {
                                  return temp2d[i][j]
                                }
                                return minTemp
                              }
                              
                              const t11 = getTemp(i1, j1) // 좌하
                              const t21 = getTemp(i2, j1) // 우하
                              const t12 = getTemp(i1, j2) // 좌상
                              const t22 = getTemp(i2, j2) // 우상
                              
                              // 보간 가중치 계산
                              const r1 = r_mm[i1]
                              const r2 = r_mm[i2]
                              const z1 = z_nm[j1]
                              const z2 = z_nm[j2]
                              
                              const dr = (r2 !== r1) ? (r - r1) / (r2 - r1) : 0
                              const dz = (z2 !== z1) ? (z - z1) / (z2 - z1) : 0
                              
                              // 양선형 보간
                              const t1 = t11 * (1 - dr) + t21 * dr // 하단 보간
                              const t2 = t12 * (1 - dr) + t22 * dr // 상단 보간
                              const temp = t1 * (1 - dz) + t2 * dz // 최종 보간
                              
                              return temp
                            }
                            
                            // 픽셀 단위로 부드러운 히트맵 그리기
                            const pixelSize = 2 // 픽셀 크기 (성능과 품질의 균형)
                            
                            for (let px = 0; px < plotWidth; px += pixelSize) {
                              for (let py = 0; py < plotHeight; py += pixelSize) {
                                // 픽셀 좌표를 물리 좌표로 변환
                                const r = minR + (px / plotWidth) * (maxR - minR)
                                const z = minZ + ((plotHeight - py) / plotHeight) * (maxZ - minZ)
                                
                                // 양선형 보간으로 온도 계산
                                const temp = bilinearInterpolate(r, z)
                                ctx.fillStyle = getColor(temp)
                                
                                // 픽셀 그리기
                                ctx.fillRect(
                                  padding.left + px,
                                  padding.top + py,
                                  pixelSize,
                                  pixelSize
                                )
                              }
                            }
                            
                            // 축 그리기
                            ctx.strokeStyle = '#333'
                            ctx.lineWidth = 2
                            ctx.beginPath()
                            // x축 (r)
                            ctx.moveTo(padding.left, padding.top + plotHeight)
                            ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight)
                            // y축 (z)
                            ctx.moveTo(padding.left, padding.top)
                            ctx.lineTo(padding.left, padding.top + plotHeight)
                            ctx.stroke()
                            
                            // 축 라벨
                            ctx.fillStyle = '#333'
                            ctx.font = 'bold 14px Arial'
                            ctx.textAlign = 'center'
                            ctx.fillText('r (mm)', padding.left + plotWidth / 2, height - 15)
                            
                            ctx.save()
                            ctx.translate(20, padding.top + plotHeight / 2)
                            ctx.rotate(-Math.PI / 2)
                            ctx.fillText('z (nm, 활성층 기준)', 0, 0)
                            ctx.restore()
                            
                            // 눈금 표시
                            ctx.font = '10px Arial'
                            ctx.strokeStyle = '#666'
                            ctx.lineWidth = 1
                            
                            // x축 눈금 (r)
                            const numTicksX = 5
                            for (let i = 0; i <= numTicksX; i++) {
                              const rValue = minR + (i / numTicksX) * (maxR - minR)
                              const x = padding.left + (rValue - minR) * scaleX
                              
                              ctx.beginPath()
                              ctx.moveTo(x, padding.top + plotHeight)
                              ctx.lineTo(x, padding.top + plotHeight + 5)
                              ctx.stroke()
                              
                              ctx.textAlign = 'center'
                              ctx.fillText(rValue.toFixed(2), x, padding.top + plotHeight + 20)
                            }
                            
                            // y축 눈금 (z)
                            const numTicksY = 5
                            for (let i = 0; i <= numTicksY; i++) {
                              const zValue = minZ + (i / numTicksY) * (maxZ - minZ)
                              const y = padding.top + plotHeight - (zValue - minZ) * scaleY
                              
                              ctx.beginPath()
                              ctx.moveTo(padding.left, y)
                              ctx.lineTo(padding.left - 5, y)
                              ctx.stroke()
                              
                              ctx.textAlign = 'right'
                              ctx.fillText(zValue.toFixed(0), padding.left - 8, y + 4)
                            }
                            
                            // 소자 반지름 수직선 표시
                            if (simulationResult.device_radius_mm) {
                              const deviceRadius = simulationResult.device_radius_mm
                              if (deviceRadius >= minR && deviceRadius <= maxR) {
                                const x = padding.left + (deviceRadius - minR) * scaleX
                                ctx.strokeStyle = '#000'
                                ctx.lineWidth = 2
                                ctx.setLineDash([5, 5])
                                ctx.beginPath()
                                ctx.moveTo(x, padding.top)
                                ctx.lineTo(x, padding.top + plotHeight)
                                ctx.stroke()
                                ctx.setLineDash([])
                                
                                // 라벨
                                ctx.fillStyle = '#000'
                                ctx.font = 'bold 11px Arial'
                                ctx.textAlign = 'left'
                                ctx.fillText('소자 반지름', x + 5, padding.top + 15)
                              }
                            }
                            
                            // 페로브스카이트 레이어 위치 수평선 표시
                            if (simulationResult.layer_names && simulationResult.layer_boundaries_nm) {
                              const layerNames = simulationResult.layer_names
                              const layerBoundaries = simulationResult.layer_boundaries_nm
                              
                              // 페로브스카이트 레이어 인덱스 찾기
                              const perovskiteIndex = layerNames.findIndex(name => 
                                name.toLowerCase().includes('perovskite')
                              )
                              
                              if (perovskiteIndex >= 0 && perovskiteIndex < layerBoundaries.length - 1) {
                                // 페로브스카이트 레이어의 시작과 끝 경계
                                const perovskiteStartZ = layerBoundaries[perovskiteIndex]
                                const perovskiteEndZ = layerBoundaries[perovskiteIndex + 1]
                                
                                // z 좌표 범위 내에 있는지 확인
                                if (perovskiteStartZ >= minZ && perovskiteStartZ <= maxZ) {
                                  const yStart = padding.top + plotHeight - (perovskiteStartZ - minZ) * scaleY
                                  
                                  ctx.strokeStyle = '#ffffff' // 하얀색
                                  ctx.lineWidth = 2
                                  ctx.setLineDash([5, 5])
                                  ctx.beginPath()
                                  ctx.moveTo(padding.left, yStart)
                                  ctx.lineTo(padding.left + plotWidth, yStart)
                                  ctx.stroke()
                                  ctx.setLineDash([])
                                  
                                  // 라벨 (하얀색 배경에 검은색 테두리로 가독성 향상)
                                  ctx.fillStyle = '#ffffff'
                                  ctx.font = 'bold 11px Arial'
                                  ctx.strokeStyle = '#000000'
                                  ctx.lineWidth = 3
                                  ctx.textAlign = 'left'
                                  ctx.strokeText('Perovskite 시작', padding.left + 5, yStart - 5)
                                  ctx.fillText('Perovskite 시작', padding.left + 5, yStart - 5)
                                }
                                
                                if (perovskiteEndZ >= minZ && perovskiteEndZ <= maxZ) {
                                  const yEnd = padding.top + plotHeight - (perovskiteEndZ - minZ) * scaleY
                                  
                                  ctx.strokeStyle = '#ffffff' // 하얀색
                                  ctx.lineWidth = 2
                                  ctx.setLineDash([5, 5])
                                  ctx.beginPath()
                                  ctx.moveTo(padding.left, yEnd)
                                  ctx.lineTo(padding.left + plotWidth, yEnd)
                                  ctx.stroke()
                                  ctx.setLineDash([])
                                  
                                  // 라벨 (하얀색 배경에 검은색 테두리로 가독성 향상)
                                  ctx.fillStyle = '#ffffff'
                                  ctx.font = 'bold 11px Arial'
                                  ctx.strokeStyle = '#000000'
                                  ctx.lineWidth = 3
                                  ctx.textAlign = 'left'
                                  ctx.strokeText('Perovskite 끝', padding.left + 5, yEnd + 15)
                                  ctx.fillText('Perovskite 끝', padding.left + 5, yEnd + 15)
                                }
                              }
                            }
                          }}
                        />
                      </div>
                      {/* 컬러바 */}
                      <div style={{
                        width: '50px',
                        height: '500px',
                        position: 'relative',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        backgroundColor: '#fff'
                      }}>
                        <canvas
                          id="rz-colorbar-canvas"
                          style={{
                            width: '100%',
                            height: '100%',
                            display: 'block'
                          }}
                          ref={(canvas) => {
                            if (!canvas || !simulationResult.temperature_2d) return
                            
                            const ctx = canvas.getContext('2d')
                            // 고해상도 디스플레이 지원
                            const dpr = window.devicePixelRatio || 1
                            const width = canvas.offsetWidth
                            const height = canvas.offsetHeight
                            canvas.width = width * dpr
                            canvas.height = height * dpr
                            ctx.scale(dpr, dpr)
                            canvas.style.width = width + 'px'
                            canvas.style.height = height + 'px'
                            
                            const temp2d = simulationResult.temperature_2d
                            const allTemps = temp2d.flat()
                            const minTemp = Math.min(...allTemps)
                            const maxTemp = Math.max(...allTemps)
                            const tempRange = maxTemp - minTemp
                            
                            // 자연스러운 색상 맵핑 함수 (히트맵과 동일)
                            const getColor = (temp) => {
                              const ratio = Math.max(0, Math.min(1, (temp - minTemp) / tempRange))
                              
                              // 더 부드러운 색상 전환을 위한 커브 적용
                              const smoothRatio = ratio * ratio * (3 - 2 * ratio) // smoothstep 함수
                              
                              // 색상 단계별 정의 (더 자연스러운 그라데이션)
                              let r, g, b
                              
                              if (smoothRatio < 0.25) {
                                // 파란색 -> 청록색 (0 ~ 0.25)
                                const t = smoothRatio / 0.25
                                r = 0
                                g = Math.floor(100 + t * 155)
                                b = Math.floor(200 + t * 55)
                              } else if (smoothRatio < 0.5) {
                                // 청록색 -> 녹색 -> 노란색 (0.25 ~ 0.5)
                                const t = (smoothRatio - 0.25) / 0.25
                                r = Math.floor(0 + t * 255)
                                g = Math.floor(255)
                                b = Math.floor(255 - t * 255)
                              } else if (smoothRatio < 0.75) {
                                // 노란색 -> 주황색 (0.5 ~ 0.75)
                                const t = (smoothRatio - 0.5) / 0.25
                                r = Math.floor(255)
                                g = Math.floor(255 - t * 100)
                                b = 0
                              } else {
                                // 주황색 -> 빨간색 (0.75 ~ 1.0)
                                const t = (smoothRatio - 0.75) / 0.25
                                r = Math.floor(255)
                                g = Math.floor(155 - t * 155)
                                b = 0
                              }
                              
                              return `rgb(${r}, ${g}, ${b})`
                            }
                            
                            // 그라데이션 그리기
                            const numSteps = 100
                            for (let i = 0; i < numSteps; i++) {
                              const ratio = i / numSteps
                              const temp = minTemp + ratio * tempRange
                              ctx.fillStyle = getColor(temp)
                              const y = height - (i / numSteps) * height
                              const stepHeight = height / numSteps
                              ctx.fillRect(0, y, width, stepHeight)
                            }
                            
                            // 라벨 추가
                            ctx.fillStyle = '#333'
                            ctx.font = '10px Arial'
                            ctx.textAlign = 'center'
                            ctx.save()
                            ctx.translate(width / 2, 10)
                            ctx.fillText(maxTemp.toFixed(1) + '°C', 0, 0)
                            ctx.translate(0, height - 20)
                            ctx.fillText(minTemp.toFixed(1) + '°C', 0, 0)
                            ctx.restore()
                          }}
                        />
                      </div>
                    </div>
                    <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#666' }}>
                      * r-z 평면 히트맵: 가로축은 r (mm), 세로축은 z (nm, 활성층 기준). 파란색은 낮은 온도, 빨간색은 높은 온도를 나타냅니다. 검은색 점선은 소자 반지름 위치, 하얀색 점선은 페로브스카이트 레이어 경계를 나타냅니다.
                    </div>
                  </div>
                </div>
              )}

              {/* 최종 온도 프로파일 (r=0에서의 z 방향) */}
              <div className="chart-container" ref={chart1Ref} style={{ position: 'relative' }}>
                <h3 style={{ marginBottom: '60px' }}>
                  온도 프로파일 (r=0, t = {simulationResult.time && simulationResult.time.length > 0 ? simulationResult.time[simulationResult.time.length - 1]?.toFixed(1) || 'N/A' : 'N/A'} s)
                </h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="position" 
                      type="number"
                      label={{ value: 'ITO/Glass 경계로부터의 위치 (nm)', position: 'insideBottom', offset: -5 }}
                      domain={['dataMin', 'dataMax']}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      tickFormatter={(value) => {
                        if (Math.abs(value) >= 1e9) return (value / 1e9).toFixed(2) + 'G'
                        if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(2) + 'M'
                        if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(2) + 'k'
                        return value.toFixed(0)
                      }}
                    />
                    <YAxis 
                      label={{ value: '온도 (°C)', angle: -90, position: 'insideLeft' }}
                      domain={['auto', 'auto']}
                      allowDataOverflow={false}
                      tick={{ angle: -30, textAnchor: 'end' }}
                    />
                    <Tooltip />
                    {/* 레이어 영역 표시 */}
                    {getLayerAreas().map((area, idx) => {
                      // x1, x2가 유효한 범위 내에 있는지 확인
                      const activeProfile = getActiveProfile()
                      if (activeProfile.length === 0) return null
                      const minX = Math.min(...activeProfile.map(p => p.position))
                      const maxX = Math.max(...activeProfile.map(p => p.position))
                      
                      // 영역이 그래프 범위 내에 있는 경우만 표시
                      if (area.x2 < minX || area.x1 > maxX) return null
                      
                      // 영역을 그래프 범위로 제한
                      const x1 = Math.max(area.x1, minX)
                      const x2 = Math.min(area.x2, maxX)
                      
                      return (
                        <ReferenceArea
                          key={`area-${idx}`}
                          x1={x1}
                          x2={x2}
                          fill={area.color}
                          fillOpacity={0.35}
                          stroke={area.color}
                          strokeOpacity={0.5}
                          strokeWidth={1}
                        />
                      )
                    })}
                    {/* 레이어 경계 수직선 - ITO부터 Cathode까지 */}
                    {(() => {
                      const { layer_boundaries_nm } = simulationResult
                      if (!layer_boundaries_nm || layer_boundaries_nm.length === 0) return null
                      
                      // 활성층 레이어 경계선 (ITO 시작점을 0으로 재조정됨)
                      const boundaries = layer_boundaries_nm.slice(1) // 첫 번째는 0이므로 제외
                      
                      return boundaries.map((boundary, idx) => (
                        <ReferenceLine
                          key={`line-${idx}`}
                          x={boundary}
                          stroke="#888"
                          strokeDasharray="3 3"
                          strokeOpacity={0.5}
                        />
                      ))
                    })()}
                    {/* Glass 영역은 ITO~Cathode 그래프에서 제외 */}
                    <Line 
                      data={getActiveProfile()}
                      type="monotone" 
                      dataKey="temperature" 
                      stroke="#2563eb" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
                {/* 레이어 라벨 오버레이 - ITO~Cathode 영역만 표시하므로 제거 */}
              </div>

              {/* 시뮬레이션 요약 정보 */}
              {(() => {
                const stats = getSimulationStats()
                if (!stats) return null
                
                // 가장 위쪽 레이어 이름 찾기
                const topLayerName = simulationResult.layer_names && simulationResult.layer_names.length > 0
                  ? simulationResult.layer_names[simulationResult.layer_names.length - 1]
                  : '최상단 레이어'
                
                return (
                  <div style={{
                    marginBottom: '30px',
                    padding: '20px',
                    backgroundColor: '#f8f9fa',
                    borderRadius: '8px',
                    border: '1px solid #dee2e6'
                  }}>
                    <h3 style={{ marginTop: 0, marginBottom: '15px' }}>시뮬레이션 요약</h3>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                      gap: '15px'
                    }}>
                      <div>
                        <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>하단 끝 온도 (Glass 시작점)</div>
                        <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#2563eb' }}>
                          {stats.bottomTemp !== null && stats.bottomTemp !== undefined ? `${stats.bottomTemp.toFixed(2)} °C` : 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>상단 끝 온도 ({topLayerName} 끝)</div>
                        <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#dc2626' }}>
                          {stats.topTemp !== null && stats.topTemp !== undefined ? `${stats.topTemp.toFixed(2)} °C` : 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>페로브스카이트 중간 온도</div>
                        <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#16a34a' }}>
                          {stats.finalTemp !== null && stats.finalTemp !== undefined ? `${stats.finalTemp.toFixed(2)} °C` : 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>소자 내부 온도 차이 (Vertical)</div>
                        <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#9333ea' }}>
                          {stats.tempDifference !== null && stats.tempDifference !== undefined ? `${stats.tempDifference.toFixed(2)} °C` : 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>소자 내부 온도 차이 (Lateral)</div>
                        <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#f59e0b' }}>
                          {stats.tempDifferenceLateral !== null && stats.tempDifferenceLateral !== undefined ? `${stats.tempDifferenceLateral.toFixed(2)} °C` : 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })()}

              {/* 페로브스카이트 중간 지점의 시간에 따른 온도 */}
              <div className="chart-container" ref={chart2Ref}>
                <h3>페로브스카이트 중간 지점의 시간에 따른 온도</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={getPerovskiteCenterProfile()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      type="number"
                      label={{ value: '시간 (s)', position: 'insideBottom', offset: -5 }}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis 
                      label={{ value: '온도 (°C)', angle: -90, position: 'insideLeft' }}
                      domain={['auto', 'auto']}
                      allowDataOverflow={false}
                    />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="temperature" 
                      stroke="#16a34a" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* 저장 및 내보내기 버튼 */}
              <div style={{ marginTop: '30px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button
                  onClick={handleSaveExcel}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#333',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.95em',
                    fontWeight: '600',
                    transition: 'all 0.3s ease',
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#555'
                    e.currentTarget.style.transform = 'translateY(-2px)'
                    e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#333'
                    e.currentTarget.style.transform = 'translateY(0)'
                    e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)'
                  }}
                >
                  Excel 저장
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
