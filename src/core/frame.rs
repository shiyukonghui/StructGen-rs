use serde::{Deserialize, Serialize};

use super::error::CoreError;

/// 精确可表示为 f64 的最大 i64 绝对值 (2^53 = 9007199254740992)
const EXACT_F64_MAX_I64: i64 = 1i64 << 53;

/// 单个状态值的标记联合体，统一承载整型、浮点型和布尔型数据
///
/// # 不变量
/// `Float` 变体必须持有有限值（非 NaN、非 ±Infinity）。
/// 外部构造应使用 [`FrameState::float`] 或 [`FrameState::float_or_zero`] 方法。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameState {
    /// 有符号 64 位整数（可表示离散状态、符号索引等）
    Integer(i64),
    /// 64 位浮点数（连续系统的状态变量）
    ///
    /// # 不变量
    /// 该变体必须持有有限值（非 NaN、非 ±Infinity）。
    /// 请使用 [`FrameState::float`] 或 [`FrameState::float_or_zero`] 构造以确保该约束。
    Float(f64),
    /// 布尔值（二值网格元胞等）
    Bool(bool),
}

impl FrameState {
    /// 创建有限浮点数值，拒绝 NaN 和 ±Infinity
    ///
    /// # Errors
    /// 当值为非有限数（NaN 或 ±Infinity）时返回 [`CoreError::InvalidParams`]
    pub fn float(value: f64) -> Result<Self, CoreError> {
        if !value.is_finite() {
            return Err(CoreError::InvalidParams(
                "Float value must be finite (not NaN or Infinity)".into(),
            ));
        }
        Ok(FrameState::Float(value))
    }

    /// 创建浮点数值，非有限值（NaN、±Infinity）自动替换为 0.0
    ///
    /// 适用于生成器流式输出场景：数值发散时不中断流，而是用 0.0 占位。
    pub fn float_or_zero(value: f64) -> Self {
        if value.is_finite() {
            FrameState::Float(value)
        } else {
            FrameState::Float(0.0)
        }
    }

    /// 尝试将值解释为 i64
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            FrameState::Integer(v) => Some(*v),
            FrameState::Float(_) => None,
            FrameState::Bool(v) => Some(*v as i64),
        }
    }

    /// 尝试将值解释为 f64
    ///
    /// 对于 `Integer` 变体，仅当值在 `[-2^53, 2^53]` 范围内时返回 `Some`，
    /// 超出此范围的整数无法精确表示为 f64，返回 `None`。
    pub fn as_float(&self) -> Option<f64> {
        match self {
            FrameState::Integer(v) => {
                if v.abs() <= EXACT_F64_MAX_I64 {
                    Some(*v as f64)
                } else {
                    None
                }
            }
            FrameState::Float(v) => Some(*v),
            FrameState::Bool(v) => Some(*v as u8 as f64),
        }
    }

    /// 尝试将值解释为 bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FrameState::Integer(v) => Some(*v != 0),
            FrameState::Float(_) => None,
            FrameState::Bool(v) => Some(*v),
        }
    }

    /// 返回变体的判别名
    pub fn variant_name(&self) -> &'static str {
        match self {
            FrameState::Integer(_) => "Integer",
            FrameState::Float(_) => "Float",
            FrameState::Bool(_) => "Bool",
        }
    }
}

/// 一帧中所有状态值的集合，即一个样本在单个时间步上的完整状态快照
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameData {
    /// 状态值序列，顺序与生成器定义的状态维度一致
    pub values: Vec<FrameState>,
}

impl FrameData {
    /// 创建空的帧数据
    pub fn new() -> Self {
        FrameData { values: Vec::new() }
    }

    /// 状态维度（values 长度）
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// 判断是否为空帧
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for FrameData {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<FrameState> for FrameData {
    fn from_iter<I: IntoIterator<Item = FrameState>>(iter: I) -> Self {
        FrameData {
            values: iter.into_iter().collect(),
        }
    }
}

/// 一个时间步的完整帧，包含步索引、状态数据和可选的语义标签
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceFrame {
    /// 时间步索引，从 0 开始递增
    pub step_index: u64,
    /// 该时间步的完整状态快照
    pub state: FrameData,
    /// 可选的语义标签
    pub label: Option<String>,
}

impl SequenceFrame {
    /// 创建无标签的帧
    pub fn new(step_index: u64, state: FrameData) -> Self {
        SequenceFrame {
            step_index,
            state,
            label: None,
        }
    }

    /// 创建带标签的帧
    pub fn with_label(step_index: u64, state: FrameData, label: impl Into<String>) -> Self {
        SequenceFrame {
            step_index,
            state,
            label: Some(label.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_state_conversions() {
        let int_val = FrameState::Integer(42);
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(int_val.as_bool(), Some(true));

        let float_val = FrameState::Float(3.14);
        assert_eq!(float_val.as_integer(), None);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_bool(), None);

        let bool_val = FrameState::Bool(true);
        assert_eq!(bool_val.as_integer(), Some(1));
        assert_eq!(bool_val.as_bool(), Some(true));
    }

    #[test]
    fn test_frame_state_zero_integer_is_false() {
        let zero = FrameState::Integer(0);
        assert_eq!(zero.as_bool(), Some(false));
    }

    #[test]
    fn test_variant_name() {
        assert_eq!(FrameState::Integer(0).variant_name(), "Integer");
        assert_eq!(FrameState::Float(0.0).variant_name(), "Float");
        assert_eq!(FrameState::Bool(false).variant_name(), "Bool");
    }

    #[test]
    fn test_frame_data_construction() {
        let data: FrameData = [FrameState::Integer(1), FrameState::Bool(true)]
            .into_iter()
            .collect();
        assert_eq!(data.dim(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_frame_data_empty() {
        let data = FrameData::new();
        assert_eq!(data.dim(), 0);
        assert!(data.is_empty());
    }

    #[test]
    fn test_frame_data_default() {
        let data = FrameData::default();
        assert!(data.is_empty());
    }

    #[test]
    fn test_frame_data_from_iterator() {
        let data: FrameData = std::iter::repeat_with(|| FrameState::Integer(1))
            .take(5)
            .collect();
        assert_eq!(data.dim(), 5);
    }

    #[test]
    fn test_sequence_frame_new() {
        let data: FrameData = [FrameState::Integer(5)].into_iter().collect();
        let frame = SequenceFrame::new(0, data.clone());
        assert_eq!(frame.step_index, 0);
        assert_eq!(frame.state, data);
        assert_eq!(frame.label, None);
    }

    #[test]
    fn test_sequence_frame_with_label() {
        let data = FrameData::new();
        let frame = SequenceFrame::with_label(1, data.clone(), "period_2");
        assert_eq!(frame.step_index, 1);
        assert_eq!(frame.label, Some("period_2".to_string()));
    }

    #[test]
    fn test_frame_state_serialization() {
        let state = FrameState::Integer(123);
        let json = serde_json::to_string(&state).unwrap();
        let restored: FrameState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn test_sequence_frame_serialization() {
        let data: FrameData = [FrameState::Integer(1), FrameState::Bool(true)]
            .into_iter()
            .collect();
        let frame = SequenceFrame::with_label(0, data, "test");
        let json = serde_json::to_string(&frame).unwrap();
        let restored: SequenceFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.step_index, frame.step_index);
        assert_eq!(restored.state, frame.state);
        assert_eq!(restored.label, frame.label);
    }

    // --- 新增边界条件测试 ---

    #[test]
    fn test_float_checked_constructor_finite() {
        let state = FrameState::float(3.14).unwrap();
        assert_eq!(state, FrameState::Float(3.14));
    }

    #[test]
    fn test_float_checked_constructor_nan() {
        let result = FrameState::float(f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_float_checked_constructor_infinity() {
        let result = FrameState::float(f64::INFINITY);
        assert!(result.is_err());
        let result = FrameState::float(f64::NEG_INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn test_as_float_large_integer_out_of_range() {
        let large = FrameState::Integer((1i64 << 53) + 1);
        assert_eq!(large.as_float(), None);
    }

    #[test]
    fn test_as_float_integer_at_boundary() {
        let boundary = FrameState::Integer(1i64 << 53);
        assert_eq!(boundary.as_float(), Some((1i64 << 53) as f64));
    }

    #[test]
    fn test_as_float_negative_large_integer() {
        let large = FrameState::Integer(-((1i64 << 53) + 1));
        assert_eq!(large.as_float(), None);
    }

    #[test]
    fn test_as_float_negative_at_boundary() {
        let boundary = FrameState::Integer(-(1i64 << 53));
        assert_eq!(boundary.as_float(), Some(-(1i64 << 53) as f64));
    }

    #[test]
    fn test_bool_as_integer() {
        assert_eq!(FrameState::Bool(true).as_integer(), Some(1));
        assert_eq!(FrameState::Bool(false).as_integer(), Some(0));
    }

    #[test]
    fn test_bool_as_float() {
        assert_eq!(FrameState::Bool(true).as_float(), Some(1.0));
        assert_eq!(FrameState::Bool(false).as_float(), Some(0.0));
    }

    #[test]
    fn test_float_or_zero_finite() {
        assert_eq!(FrameState::float_or_zero(3.14), FrameState::Float(3.14));
    }

    #[test]
    fn test_float_or_zero_nan() {
        assert_eq!(FrameState::float_or_zero(f64::NAN), FrameState::Float(0.0));
    }

    #[test]
    fn test_float_or_zero_infinity() {
        assert_eq!(FrameState::float_or_zero(f64::INFINITY), FrameState::Float(0.0));
        assert_eq!(FrameState::float_or_zero(f64::NEG_INFINITY), FrameState::Float(0.0));
    }

    #[test]
    fn test_float_or_zero_zero() {
        assert_eq!(FrameState::float_or_zero(0.0), FrameState::Float(0.0));
    }

    #[test]
    fn test_float_nan_serialization_produces_null() {
        // 直接构造 NaN（绕过 float() 检查）时，serde_json 将 NaN 序列化为 null
        let state = FrameState::Float(f64::NAN);
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("null"), "NaN should serialize to null: {}", json);
    }

    #[test]
    fn test_float_nan_roundtrip_fails() {
        // NaN → JSON(null) → 反序列化失败：null 无法转为 f64
        let state = FrameState::Float(f64::NAN);
        let json = serde_json::to_string(&state).unwrap();
        let result: Result<FrameState, _> = serde_json::from_str(&json);
        assert!(result.is_err(), "NaN round-trip should fail");
    }

    #[test]
    fn test_float_infinity_roundtrip_fails() {
        let state = FrameState::Float(f64::INFINITY);
        let json = serde_json::to_string(&state).unwrap();
        let result: Result<FrameState, _> = serde_json::from_str(&json);
        assert!(result.is_err(), "Infinity round-trip should fail");
    }
}
