"""
Phase A Handover Validator.

Verifies that Phase A outputs are complete and valid for Phase D ingestion.

Handover Contract:
- clean_dataset.arrow exists with post_masked column
- projection_matrix.pt exists and is idempotent
- pollution_logs.arrow exists

Implements: phaseA-D_implementation_plan.md Section 9.3
"""

import logging
from pathlib import Path
import torch
import pyarrow.feather as feather

logger = logging.getLogger(__name__)


def validate_handover(output_dir: Path) -> bool:
    """
    Validate Phase A handover artifacts.
    
    Args:
        output_dir: Phase A output directory.
        
    Returns:
        True if validation passes, False otherwise.
    """
    output_dir = Path(output_dir)
    
    logger.info("=" * 80)
    logger.info("PHASE A HANDOVER VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    
    all_passed = True
    
    # Check 1: clean_dataset.arrow exists
    clean_dataset_path = output_dir / "clean_dataset.arrow"
    if not clean_dataset_path.exists():
        logger.error(f"✗ Clean dataset not found: {clean_dataset_path}")
        all_passed = False
    else:
        logger.info(f"✓ Clean dataset exists: {clean_dataset_path}")
        
        # Verify schema
        try:
            table = feather.read_table(clean_dataset_path)
            
            # Check for post_masked column
            if "post_masked" not in table.column_names:
                logger.error("✗ Clean dataset missing 'post_masked' column")
                all_passed = False
            else:
                logger.info("✓ Clean dataset has 'post_masked' column")
                
                # Check that post_masked is populated
                post_masked = table["post_masked"].to_pylist()
                if all(text is None or text == "" for text in post_masked):
                    logger.warning("⚠ All post_masked values are empty")
                else:
                    num_masked = sum(1 for text in post_masked if "[MASK:" in text)
                    logger.info(f"✓ {num_masked}/{len(post_masked)} posts have masks")
        
        except Exception as e:
            logger.error(f"✗ Failed to load clean dataset: {e}")
            all_passed = False
    
    # Check 2: projection_matrix.pt exists and is valid
    projection_matrix_path = output_dir / "projection_matrix.pt"
    if not projection_matrix_path.exists():
        logger.error(f"✗ Projection matrix not found: {projection_matrix_path}")
        all_passed = False
    else:
        logger.info(f"✓ Projection matrix exists: {projection_matrix_path}")
        
        # Verify projection matrix
        try:
            P = torch.load(projection_matrix_path)
            
            # Check shape (should be square)
            if P.shape[0] != P.shape[1]:
                logger.error(f"✗ Projection matrix not square: {P.shape}")
                all_passed = False
            else:
                logger.info(f"✓ Projection matrix is square: {P.shape}")
            
            # Check idempotence: P^2 ≈ P
            P_squared = P @ P
            error = torch.abs(P_squared - P).max().item()
            
            if error > 1e-4:
                logger.error(f"✗ Projection matrix not idempotent: error={error:.2e}")
                all_passed = False
            else:
                logger.info(f"✓ Projection matrix is idempotent: error={error:.2e}")
        
        except Exception as e:
            logger.error(f"✗ Failed to load projection matrix: {e}")
            all_passed = False
    
    # Check 3: pollution_logs.arrow exists
    pollution_logs_path = output_dir / "pollution_logs.arrow"
    if not pollution_logs_path.exists():
        logger.error(f"✗ Pollution logs not found: {pollution_logs_path}")
        all_passed = False
    else:
        logger.info(f"✓ Pollution logs exist: {pollution_logs_path}")
        
        # Verify pollution logs
        try:
            logs_table = feather.read_table(pollution_logs_path)
            
            required_cols = [
                "post_id", "span_start", "span_end", "span_text",
                "entity_type", "confidence", "mask_token"
            ]
            
            missing_cols = [col for col in required_cols if col not in logs_table.column_names]
            if missing_cols:
                logger.error(f"✗ Pollution logs missing columns: {missing_cols}")
                all_passed = False
            else:
                logger.info(f"✓ Pollution logs have all required columns")
                logger.info(f"  Total pollution spans logged: {len(logs_table)}")
        
        except Exception as e:
            logger.error(f"✗ Failed to load pollution logs: {e}")
            all_passed = False
    
    # Summary
    logger.info("=" * 80)
    if all_passed:
        logger.info("✓ HANDOVER VALIDATION PASSED")
        logger.info("  Phase A outputs are ready for Phase D ingestion")
    else:
        logger.error("✗ HANDOVER VALIDATION FAILED")
        logger.error("  Please fix errors before proceeding to Phase D")
    logger.info("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_phase_a_handover.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    success = validate_handover(output_dir)
    
    sys.exit(0 if success else 1)
