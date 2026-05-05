pub mod boolean_network;
pub mod ca;
pub mod ca2d;
pub mod ca3d;
pub mod ca_common;
pub mod ca_rules;
pub mod formal_grammar;
pub mod ifs;
pub mod logistic;
pub mod lorenz;
pub mod lsystem;
pub mod nbody;
pub mod nca2d;
pub mod vm;

mod rng;

use crate::core::registry::GeneratorRegistry;
use crate::core::CoreError;

/// 向注册表中注册所有内置生成器
///
/// # Errors
/// 当注册失败时返回错误（如名称冲突）
pub fn register_all(registry: &mut GeneratorRegistry) -> Result<(), CoreError> {
    registry.register("ca", ca::ca_factory)?;
    registry.register("cellular_automaton", ca::ca_factory)?;
    registry.register("ca2d", ca2d::ca2d_factory)?;
    registry.register("cellular_automaton_2d", ca2d::ca2d_factory)?;
    registry.register("ca3d", ca3d::ca3d_factory)?;
    registry.register("cellular_automaton_3d", ca3d::ca3d_factory)?;
    registry.register("nca2d", nca2d::nca2d_factory)?;
    registry.register("neural_cellular_automaton_2d", nca2d::nca2d_factory)?;
    registry.register("lorenz", lorenz::lorenz_factory)?;
    registry.register("lorenz_system", lorenz::lorenz_factory)?;
    registry.register("logistic", logistic::logistic_factory)?;
    registry.register("logistic_map", logistic::logistic_factory)?;
    registry.register("nbody", nbody::nbody_factory)?;
    registry.register("nbody_sim", nbody::nbody_factory)?;
    registry.register("lsystem", lsystem::lsystem_factory)?;
    registry.register("vm", vm::vm_factory)?;
    registry.register("algorithm_vm", vm::vm_factory)?;
    registry.register("boolean_network", boolean_network::boolean_network_factory)?;
    registry.register("ifs", ifs::ifs_factory)?;
    registry.register("formal_grammar", formal_grammar::formal_grammar_factory)?;
    Ok(())
}
