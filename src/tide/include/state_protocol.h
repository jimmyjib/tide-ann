#pragma once
/**
 * State-passing protocol
 *
 * Main include file that pulls in all protocol definitions.
 * Backward-compatible: existing code can still include "state_protocol.h"
 */

// Include all protocol sub-headers
#include "protocol/message_types.h"
#include "protocol/search_protocol.h"
#include "protocol/insert_protocol.h"
#include "protocol/broadcast_messages.h"
