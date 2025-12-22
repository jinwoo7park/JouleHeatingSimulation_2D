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
  ReferenceLine
} from 'recharts'
import * as XLSX from 'xlsx'

const LAYER_NAMES = ['Glass', 'ITO', 'HTL', 'Perovskite', 'ETL', 'Cathode', 'Heat spreader', 'Heat sink']
// 각 레이어의 두께 입력 단위: Glass(mm), Heat spreader(μm), Heat sink(mm), 나머지(nm)
const THICKNESS_UNITS = ['mm', 'nm', 'nm', 'nm', 'nm', 'nm', 'μm', 'mm']
const DEFAULT_VALUES = {
  layer_names: LAYER_NAMES,
  // Heat spreader: UV curable resin (polymer) - k=20, ρ=1100, cp=1800
  // Heat sink: Silicon - k=150, ρ=2330, cp=700
  k_therm_layers: [0.8, 10.0, 0.2, 0.5, 0.2, 200.0, 20.0, 150.0],
  rho_layers: [2500, 7140, 1000, 4100, 1200, 2700, 1100, 2330],
  c_p_layers: [1000, 280, 1500, 250, 1500, 900, 1800, 700],
  // 기본값: Glass=1.1mm, Heat spreader=3μm, Heat sink=1mm, 나머지=nm
  thickness_layers_nm: [1100000, 70, 80, 280, 50, 100, 3000, 1000000], // Heat spreader=3μm, Heat sink=1mm
  layer_enabled: [true, true, true, true, true, true, false, false], // Heat spreader와 Heat sink는 기본적으로 비활성화
  voltage: 2.9,
  current_density: 300.0,
  eqe: 0.2, // External Quantum Efficiency (20%)
  epsilon_top: 0.05,
  epsilon_bottom: 0.85,
  h_conv: 10.0,
  T_ambient: 25.0, // 섭씨 (°C)
  t_start: 0,
  t_end: 1000.0
}

function App() {
  const [logoError, setLogoError] = useState(false)
  const [formData, setFormData] = useState(DEFAULT_VALUES)
  const [simulationResult, setSimulationResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
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
    
    // Heat sink (인덱스 7)를 선택하면 Heat spreader (인덱스 6)도 자동 선택
    if (index === 7 && enabled) {
      newFormData.layer_enabled[6] = true
    }
    
    // Heat spreader (인덱스 6)를 해제하면 Heat sink (인덱스 7)도 자동 해제
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
        T_ambient: celsiusToKelvin(formData.T_ambient)
      }
      
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      })
      
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
            // traceback에서 파일명과 라인 번호 추출
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
        return
      }
      
      const data = await response.json()
      
      if (data.success) {
        // 켈빈을 섭씨로 변환하여 저장
        const convertedData = {
          ...data,
          temperature_active: data.temperature_active.map(row => 
            row.map(kelvin => kelvinToCelsius(kelvin))
          ),
          temperature_glass: data.temperature_glass.map(row => 
            row.map(kelvin => kelvinToCelsius(kelvin))
          ),
          perovskite_center_temp: data.perovskite_center_temp.map(kelvin => 
            kelvinToCelsius(kelvin)
          )
        }
        setSimulationResult(convertedData)
      } else {
        // 상세한 오류 정보 표시
        let errorMessage = data.error || '시뮬레이션 실행 중 오류가 발생했습니다.'
        if (data.error_details) {
          console.error('상세 오류 정보:', data.error_details)
          if (data.error_details.traceback) {
            const tracebackLines = data.error_details.traceback.split('\n')
            const relevantLine = tracebackLines.find(line => 
              line.includes('api/simulate.py') || line.includes('File')
            )
            if (relevantLine) {
              errorMessage += `\n\n위치: ${relevantLine.trim()}`
            }
          }
        }
        setError(errorMessage)
      }
    } catch (err) {
      console.error('API 호출 오류:', err)
      setError(`서버에 연결할 수 없습니다: ${err.message || '네트워크 오류가 발생했습니다.'}`)
    } finally {
      setLoading(false)
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

  // Glass 물결선 데이터
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
  
  // 활성층 온도 프로파일 데이터
  const getActiveProfile = () => {
    if (!simulationResult) return []
    
    const { time, position_active_nm, temperature_active } = simulationResult
    const finalTimeIndex = time.length - 1
    
    if (!position_active_nm || !temperature_active) return []
    
    return position_active_nm.map((pos, idx) => ({
      position: pos,
      temperature: temperature_active[idx][finalTimeIndex]
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
  
  // 레이어 영역 데이터 (ReferenceArea용)
  const getLayerAreas = () => {
    if (!simulationResult) return []
    
    const { layer_boundaries_nm } = simulationResult
    const areas = []
    
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
  
  // 시뮬레이션 기본 정보 계산
  const getSimulationStats = () => {
    if (!simulationResult) return null
    
    const { temperature_active, perovskite_center_temp } = simulationResult
    const finalTimeIndex = temperature_active[0].length - 1
    
    // 시작온도 (주변 온도)
    const startTemp = formData.T_ambient
    
    // 최종온도 (페로브스카이트 중간 지점의 마지막 온도)
    const finalTemp = perovskite_center_temp[finalTimeIndex]
    
    // 소자 내부 최대/최소 온도차이 (활성층의 최종 온도 프로파일에서)
    const finalActiveTemps = temperature_active.map(row => row[finalTimeIndex])
    const maxTemp = Math.max(...finalActiveTemps)
    const minTemp = Math.min(...finalActiveTemps)
    const tempDifference = maxTemp - minTemp
    
    return {
      startTemp,
      finalTemp,
      maxTemp,
      minTemp,
      tempDifference
    }
  }
  
  // Excel 저장 함수
  const handleSaveExcel = () => {
    if (!simulationResult) {
      alert('시뮬레이션 결과가 없습니다.')
      return
    }
    
    try {
      const { time, position_active_nm, temperature_active, perovskite_center_temp } = simulationResult
      const stats = getSimulationStats()
      
      // 첫 번째 시트: 위치-시간에 따른 contour plot 데이터 (transpose)
      // 각 행이 같은 시간을 나타내도록 transpose
      const contourData = []
      // 첫 번째 행: 헤더 (시간, 위치1, 위치2, ...)
      const headerRow = ['시간', ...position_active_nm.map(pos => Number(pos))]
      contourData.push(headerRow)
      
      // 각 시간별로 위치에 따른 온도 데이터
      time.forEach((t, timeIdx) => {
        const row = [
          Number(t), // 시간 (숫자로 저장)
          ...temperature_active.map(posTemps => Number(posTemps[timeIdx]))
        ]
        contourData.push(row)
      })
      
      // 두 번째 시트: 페로브스카이트 중간 지점 데이터 + 기본 정보
      const summaryData = []
      // 시뮬레이션 파라미터를 첫 번째 행부터 표시
      summaryData.push(['시뮬레이션 파라미터', '값'])
      summaryData.push(['시작 온도', Number(stats.startTemp)])
      summaryData.push(['최종 온도', Number(stats.finalTemp)])
      summaryData.push(['소자 내부 최대 온도', Number(stats.maxTemp)])
      summaryData.push(['소자 내부 최소 온도', Number(stats.minTemp)])
      summaryData.push(['소자 내부 온도 차이', Number(stats.tempDifference)])
      
      // 빈 행 추가
      summaryData.push([])
      
      // 페로브스카이트 중간 지점 데이터
      summaryData.push(['시간', '페로브스카이트 중간 지점 온도'])
      time.forEach((t, idx) => {
        summaryData.push([Number(t), Number(perovskite_center_temp[idx])])
      })
      
      // 세 번째 시트: 시뮬레이션 입력 파라미터
      const inputParamsData = []
      inputParamsData.push(['레이어 이름', '두께 (단위별)', '열전도도 (W/m·K)', '밀도 (kg/m³)', '비열 (J/kg·K)'])
      
      // 활성화된 레이어만 출력 (Heat spreader와 Heat sink가 사용되지 않는 경우 제외)
      LAYER_NAMES.forEach((name, idx) => {
        // layer_enabled가 true인 레이어만 출력
        if (formData.layer_enabled && formData.layer_enabled[idx]) {
          inputParamsData.push([
            name,
            `${convertFromNm(formData.thickness_layers_nm[idx], THICKNESS_UNITS[idx])} ${THICKNESS_UNITS[idx]}`,
            Number(formData.k_therm_layers[idx]),
            Number(formData.rho_layers[idx]),
            Number(formData.c_p_layers[idx])
          ])
        }
      })
      
      // 빈 행 추가
      inputParamsData.push([])
      
      // 전기적 파라미터
      inputParamsData.push(['전기적 파라미터', ''])
      inputParamsData.push(['전압 (V)', Number(formData.voltage)])
      inputParamsData.push(['전류 밀도 (A/m²)', Number(formData.current_density)])
      inputParamsData.push(['EQE (External Quantum Efficiency)', Number(formData.eqe)])
      
      // 빈 행 추가
      inputParamsData.push([])
      
      // 열적 파라미터
      inputParamsData.push(['열적 파라미터', ''])
      inputParamsData.push(['상부 방사율 (Cathode)', Number(formData.epsilon_top)])
      inputParamsData.push(['하부 방사율 (Glass)', Number(formData.epsilon_bottom)])
      inputParamsData.push(['대류 계수 (W/m²·K)', Number(formData.h_conv)])
      inputParamsData.push(['주변 온도 (°C)', Number(formData.T_ambient)])
      
      // 빈 행 추가
      inputParamsData.push([])
      
      // 시뮬레이션 시간
      inputParamsData.push(['시뮬레이션 시간', ''])
      inputParamsData.push(['시작 시간 (s)', Number(formData.t_start)])
      inputParamsData.push(['종료 시간 (s)', Number(formData.t_end)])
      
      // 워크북 생성
      const wb = XLSX.utils.book_new()
      
      // 첫 번째 시트: Contour Plot 데이터 (transpose)
      const ws1 = XLSX.utils.aoa_to_sheet(contourData)
      XLSX.utils.book_append_sheet(wb, ws1, '위치-시간 온도 데이터')
      
      // 두 번째 시트: 요약 데이터
      const ws2 = XLSX.utils.aoa_to_sheet(summaryData)
      XLSX.utils.book_append_sheet(wb, ws2, '페로브스카이트 온도 및 요약')
      
      // 세 번째 시트: 입력 파라미터
      const ws3 = XLSX.utils.aoa_to_sheet(inputParamsData)
      XLSX.utils.book_append_sheet(wb, ws3, '입력 파라미터')
      
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
            <h1>Joule Heating Simulation (1D)</h1>
            <p className="subtitle">Heat dissipation in PeLED operation using 1D heat equation</p>
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
                  const isHeatSpreader = index === 6
                  const isHeatSink = index === 7
                  const showCheckbox = isHeatSpreader || isHeatSink
                  
                  return (
                    <div key={index} className="layer-card" style={{ 
                      opacity: showCheckbox && !formData.layer_enabled[index] ? 0.5 : 1 
                    }}>
                      <div className="layer-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%' }}>
                          <h4>{name}</h4>
                          {showCheckbox && (
                            <label style={{ display: 'flex', alignItems: 'center', marginLeft: isHeatSpreader ? '70px' : '20px', cursor: 'pointer' }}>
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
                  <label>전류 밀도 (A/m²)</label>
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
              
              {/* 최종 온도 프로파일 */}
              <div className="chart-container" ref={chart1Ref} style={{ position: 'relative' }}>
                <h3 style={{ marginBottom: '60px' }}>최종 온도 프로파일 (t = {simulationResult.time[simulationResult.time.length - 1].toFixed(1)} s)</h3>
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
                    />
                    <YAxis 
                      label={{ value: '온도 (°C)', angle: -90, position: 'insideLeft' }}
                      domain={['auto', 'auto']}
                      allowDataOverflow={false}
                      tick={{ angle: -30, textAnchor: 'end' }}
                    />
                    <Tooltip />
                    {/* 레이어 영역 표시 */}
                    {getLayerAreas().map((area, idx) => (
                      <ReferenceArea
                        key={`area-${idx}`}
                        x1={area.x1}
                        x2={area.x2}
                        fill={area.color}
                        fillOpacity={0.35}
                        stroke={area.color}
                        strokeOpacity={0.5}
                        strokeWidth={1}
                      />
                    ))}
                    {/* 레이어 경계 수직선 */}
                    {simulationResult.layer_boundaries_nm.slice(1).map((boundary, idx) => (
                      <ReferenceLine
                        key={`line-${idx}`}
                        x={boundary}
                        stroke="#888"
                        strokeDasharray="3 3"
                        strokeOpacity={0.5}
                      />
                    ))}
                    <Line 
                      data={getGlassWavyProfile()}
                      type="monotone" 
                      dataKey="temperature" 
                      stroke="#dc2626" 
                      strokeWidth={2}
                      dot={false}
                    />
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
                {/* 레이어 라벨 오버레이 */}
                <div style={{
                  position: 'absolute',
                  top: '50px',
                  left: '60px',
                  right: '20px',
                  height: '310px',
                  pointerEvents: 'none'
                }}>
                  {/* Glass 라벨 */}
                  {(() => {
                    const tempRange = getTemperatureRange()
                    const glassProfile = getGlassWavyProfile()
                    const glassMaxTemp = glassProfile.length > 0 ? Math.max(...glassProfile.map(p => p.temperature)) : tempRange.max
                    const yPercent = ((tempRange.max - glassMaxTemp) / (tempRange.max - tempRange.min)) * 80 + 10
                    return (
                      <div style={{
                        position: 'absolute',
                        left: 'calc((100% - 60px) * (-100 - (-200)) / (580 - (-200)))',
                        top: `${Math.max(5, Math.min(95, yPercent))}%`,
                        fontSize: '12px',
                        fontWeight: 'bold',
                        color: '#333',
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        padding: '2px 6px',
                        borderRadius: '3px',
                        border: '1px solid #dc2626',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                        whiteSpace: 'nowrap'
                      }}>
                        Glass (축약)
                      </div>
                    )
                  })()}
                  {/* 활성층 레이어 라벨 */}
                  {getLayerAreas().map((area, idx) => {
                    const xMin = -200
                    const xMax = 580
                    const xPercent = ((area.centerX - xMin) / (xMax - xMin)) * 100
                    // 온도 범위에 따라 라벨 위치 자동 조정
                    const tempRange = getTemperatureRange()
                    const activeProfile = getActiveProfile()
                    // 해당 영역의 온도 값 찾기
                    const areaTemps = activeProfile
                      .filter(p => p.position >= area.x1 && p.position <= area.x2)
                      .map(p => p.temperature)
                    const areaMaxTemp = areaTemps.length > 0 ? Math.max(...areaTemps) : tempRange.max
                    // 온도를 Y 위치로 변환 (그래프 높이의 10% 여백 고려)
                    const yPercent = ((tempRange.max - areaMaxTemp) / (tempRange.max - tempRange.min)) * 80 + 10
                    return (
                      <div
                        key={`label-${idx}`}
                        style={{
                          position: 'absolute',
                          left: `${xPercent}%`,
                          transform: 'translateX(-50%)',
                          top: `${Math.max(5, Math.min(95, yPercent))}%`,
                          fontSize: '12px',
                          fontWeight: 'bold',
                          color: '#333',
                          backgroundColor: 'rgba(255, 255, 255, 0.9)',
                          padding: '2px 6px',
                          borderRadius: '3px',
                          whiteSpace: 'nowrap',
                          border: `1px solid ${area.color}`,
                          boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                        }}
                      >
                        {area.name}
                      </div>
                    )
                  })}
                </div>
              </div>

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
